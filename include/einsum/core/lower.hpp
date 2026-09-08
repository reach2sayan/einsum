#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/core/path.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <span>

// Where a Plan meets a set of extents and becomes a sequence of Eigen calls.
// Constexpr throughout, so both entry points run the one lowering.
namespace einsum::impl {

// A run of axes is one axis in disguise when its strides nest exactly: dropping
// the extent-1 axes, each stride is the next one times the next extent.  That
// is the whole test for whether Eigen can address it without a copy.
struct Collapsed {
  bool ok = false;
  index_t extent = 1;
  index_t stride = 1;
};

[[nodiscard]] constexpr Collapsed collapse(const Layout &g) noexcept {
  // An extent-1 axis moves nothing, so its stride cannot break the nesting.
  auto live = std::views::zip(g.shape, g.strides) |
              std::views::filter(
                  [](const auto &axis) { return std::get<0>(axis) != 1; });
  const Shape ext{std::from_range, live | std::views::keys};
  const Shape str{std::from_range, live | std::views::values};
  if (ext.empty()) {
    return {.ok = true, .extent = 1, .stride = 1};
  }
  const auto nested = std::views::zip(ext, str);
  const bool broken =
      std::ranges::adjacent_find(nested, [](const auto &in, const auto &out) {
        return std::get<1>(in) != std::get<1>(out) * std::get<0>(out);
      }) != std::ranges::end(nested);
  return broken ? Collapsed{}
                : Collapsed{
                      .ok = true, .extent = product(ext), .stride = str.back()};
}

// One matrix operand of one GEMM: two collapsible groups Eigen can Map, or a
// rectangle to be packed into scratch first.  transposed means the map is
// cols x rows, which is how a column-major block reaches a Map with its inner
// stride still 1 -- the one thing keeping Eigen from a temporary.
struct Slab {
  Layout batch{};
  Layout rows{};
  Layout cols{};
  bool mappable = false;
  bool transposed = false;
  index_t outer_stride = 1;
  index_t pack_offset = 0; // into the workspace, when !mappable

  // Derived rather than stored: a copy can disagree with what it copied.
  [[nodiscard]] constexpr index_t rows_n() const noexcept {
    return rows.size();
  }
  [[nodiscard]] constexpr index_t cols_n() const noexcept {
    return cols.size();
  }
};

[[nodiscard]] constexpr Slab make_slab(const Layout &batch, const Layout &rows,
                                       const Layout &cols) noexcept {
  Slab slab{.batch = batch, .rows = rows, .cols = cols};
  const index_t rows_n = slab.rows_n();
  const index_t cols_n = slab.cols_n();

  const Collapsed r = collapse(rows);
  const Collapsed c = collapse(cols);
  if (r.ok && c.ok) {
    // A degenerate side has no stride worth honouring, so the other one's is
    // the outer stride.
    if (c.stride == 1 || cols_n == 1) {
      const index_t outer = rows_n == 1 ? cols_n : r.stride;
      // Rows that overlap are not a matrix, and a reversed axis is not an
      // OuterStride; both fall through to packing.
      if (outer > 0 && (rows_n == 1 || outer >= cols_n)) {
        slab.mappable = true;
        slab.outer_stride = outer;
      }
    }
    if (!slab.mappable && (r.stride == 1 || rows_n == 1)) {
      const index_t outer = cols_n == 1 ? rows_n : c.stride;
      if (outer > 0 && (cols_n == 1 || outer >= rows_n)) {
        slab.mappable = true;
        slab.transposed = true;
        slab.outer_stride = outer;
      }
    }
  }
  return slab;
}

struct PrepGeom {
  Layout merged{}; // over Prep::merged_labels, addressing the operand itself
  Layout keep{};   // the merged axes that survive
  Layout red{};    // the merged axes that are summed away
  bool reduced = false;
  index_t reduced_offset = 0;
  Collapsed
      red_run{};  // the reduced axes as one Eigen vector, when they are one
  Layout after{}; // over Prep::labels_after: what the steps see
};

struct StepGeom {
  Slab l{};
  Slab r{};
  Slab out{};

  // The left slab is batch x m x k, the right one batch x k x n.
  [[nodiscard]] constexpr index_t m() const noexcept { return l.rows_n(); }
  [[nodiscard]] constexpr index_t n() const noexcept { return r.cols_n(); }
  [[nodiscard]] constexpr index_t k() const noexcept { return l.cols_n(); }
  [[nodiscard]] constexpr index_t batches() const noexcept {
    return l.batch.size();
  }
  // No GEMM: every "matrix" is a scalar and the whole step is the batch.
  [[nodiscard]] constexpr bool hadamard() const noexcept {
    return m() == 1 && n() == 1 && k() == 1;
  }

  // A scratch base is an offset into the workspace, otherwise the operand's
  // own pointer.
  bool l_scratch = false;
  std::uint8_t l_operand = 0;
  index_t l_offset = 0;
  bool r_scratch = false;
  std::uint8_t r_operand = 0;
  index_t r_offset = 0;
  bool out_scratch = false; // false only on the step that writes the output
  index_t out_offset = 0;
};

struct Geometry {
  FixedVec<PrepGeom, kMaxOperands> preps{};
  FixedVec<StepGeom, kMaxOperands - 1> steps{};
  // For the tests; every step already says where its two sides are.
  Path path{};
  Shape out_shape{};
  index_t scratch_elems = 0;

  // The one-operand path: no GEMM, just a strided copy into the output.
  Layout unary_src{}; // permuted into the output's axis order already
  bool unary_from_scratch = false;
  index_t unary_offset = 0;
};

// The extents each label is bound to, in operand order.
[[nodiscard]] constexpr result<BoundExtents>
bind_extents(const Plan &plan, const std::span<const Layout> inputs) noexcept {
  if (inputs.size() != plan.operand_count()) {
    return fail(errc::operand_count_mismatch);
  }
  BoundExtents bound;
  for (const auto &[input, labels] :
       std::views::zip(inputs, plan.subscripts().operands)) {
    if (input.rank() != labels.size()) {
      return fail(errc::rank_mismatch);
    }
    // What THIS operand's axes say: a diagonal has to be square in the operand
    // that walks it, even where the same label broadcasts elsewhere.
    LabelTable<bool> seen_here;
    LabelTable<index_t> extent_here;
    for (const auto &[c, extent] : std::views::zip(labels, input.shape)) {
      BoundExtent &slot = bound[c];
      if (seen_here[c]) {
        if (extent_here[c] != extent) {
          return fail(errc::extent_conflict);
        }
        continue;
      }
      seen_here[c] = true;
      extent_here[c] = extent;

      if (!slot.known) {
        slot = {.extent = extent, .known = true};
      } else if (slot.extent == extent || extent == 1) {
        // Agreed, or this operand is stretched along the axis and what is
        // already bound stands.
      } else if (slot.extent == 1) {
        // Everything so far was stretched; this operand sets the extent.
        slot.extent = extent;
      } else {
        return fail(is_broadcast_label(c) ? errc::broadcast_mismatch
                                          : errc::extent_conflict);
      }
    }
  }
  return bound;
}

// The ranks an expand() is told about.
[[nodiscard]] constexpr FixedVec<std::uint8_t, kMaxOperands>
ranks_of(const std::span<const Layout> lays) noexcept {
  return {std::from_range, lays | std::views::transform([](const Layout &lay) {
            return static_cast<std::uint8_t>(lay.rank());
          })};
}

[[nodiscard]] constexpr Shape shape_over(const Labels &labels,
                                         const BoundExtents &bound) noexcept {
  return {std::from_range, labels | std::views::transform([&](const char c) {
                             return bound[c].extent;
                           })};
}

[[nodiscard]] constexpr result<Shape>
infer_output_shape(const Plan &plan,
                   const std::span<const Layout> inputs) noexcept {
  return bind_extents(plan, inputs).transform([&](const BoundExtents &bound) {
    return shape_over(plan.output_labels(), bound);
  });
}

namespace detail {

// The axes of `wanted` inside `layout`, whose axis i carries `axis_labels[i]`.
[[nodiscard]] constexpr Layout pick(const Labels &axis_labels,
                                    const Layout &layout,
                                    const Labels &wanted) noexcept {
  const auto axes = wanted | std::views::transform([&](const char c) {
                      return index_of(axis_labels, c);
                    });
  return {.shape = {std::from_range,
                    axes | std::views::transform([&](const std::size_t a) {
                      return layout.shape[a];
                    })},
          .strides = {std::from_range,
                      axes | std::views::transform([&](const std::size_t a) {
                        return layout.strides[a];
                      })}};
}

// A running bump allocator over the workspace, each block starting on a
// multiple of 16 elements so a Map never straddles a cache line it need not.
struct Bump {
  index_t cursor = 0;
  constexpr index_t take(const index_t count) noexcept {
    const index_t at = cursor;
    cursor += (std::max<index_t>(count, 1) + 15) / 16 * 16;
    return at;
  }
};

} // namespace detail

[[nodiscard]] constexpr result<Geometry>
make_geometry(const Plan &plan, const std::span<const Layout> inputs,
              const Layout &out, const path order = path::greedy) noexcept {
  const auto bound = bind_extents(plan, inputs);
  if (!bound) {
    return std::unexpected{bound.error()};
  }

  Geometry geom;
  geom.out_shape = shape_over(plan.output_labels(), *bound);
  if (!std::ranges::equal(out.shape, geom.out_shape)) {
    return fail(errc::output_mismatch);
  }

  detail::Bump bump;

  // --- the per-operand preparation -------------------------------------------
  for (const auto &[input, prep, op_labels] : std::views::zip(
           inputs, access::preps(plan), plan.subscripts().operands)) {
    PrepGeom pg;

    // A label repeated inside one operand walks the diagonal, and the stride
    // along a diagonal is the sum of the strides of the axes it crosses.
    pg.merged.shape = shape_over(prep.merged_labels, *bound);
    pg.merged.strides = Shape(prep.merged_labels.size());
    // A stretched axis contributes no stride: every index along it reads the
    // one element there is.  collapse() refuses a zero stride, so such an
    // operand takes the packing path and no kernel knows of broadcasting.
    for (const auto &[c, extent, stride, into] : std::views::zip(
             op_labels, input.shape, input.strides, prep.merged_into)) {
      const bool stretched = extent == 1 && (*bound)[c].extent != 1;
      pg.merged.strides[into] += stretched ? index_t{0} : stride;
    }

    for (const auto &[c, extent, stride] : std::views::zip(
             prep.merged_labels, pg.merged.shape, pg.merged.strides)) {
      Layout &half = std::ranges::contains(prep.reduce, c) ? pg.red : pg.keep;
      half.shape.push_back(extent);
      half.strides.push_back(stride);
    }

    pg.reduced = !prep.reduce.empty();
    if (pg.reduced) {
      pg.red_run = collapse(pg.red);
      pg.reduced_offset = bump.take(pg.keep.size());
      pg.after = make_layout<RowMajor>(pg.keep.shape);
    } else {
      pg.after = pg.merged;
    }
    geom.preps.push_back(pg);
  }

  // What is left after the diagonal and the sum is a permutation of the output,
  // so the whole plan is one strided copy.
  if (plan.operand_count() == 1) {
    geom.unary_src = detail::pick(access::preps(plan)[0].labels_after,
                                  geom.preps[0].after, plan.output_labels());
    geom.unary_from_scratch = geom.preps[0].reduced;
    geom.unary_offset = geom.preps[0].reduced_offset;
    geom.scratch_elems = bump.cursor;
    return geom;
  }

  // Where each live tensor is: the operands, then one slot per step for the
  // intermediate it produces.  A cost-chosen path does not contract left to
  // right, so a step's sides are looked up rather than carried along.
  struct Source {
    Labels labels{};
    Layout layout{};
    bool scratch = false;
    index_t offset = 0;
    std::uint8_t operand = 0;
  };
  std::array<Source, 2 * kMaxOperands> sources{};
  FixedVec<Labels, kMaxOperands> live_labels;
  for (const auto &[i, prep, pg] : std::views::zip(
           std::views::iota(std::size_t{0}), access::preps(plan), geom.preps)) {
    sources[i] = {.labels = prep.labels_after,
                  .layout = pg.after,
                  .scratch = pg.reduced,
                  .offset = pg.reduced_offset,
                  .operand = static_cast<std::uint8_t>(i)};
    live_labels.push_back(prep.labels_after);
  }

  const auto chosen = choose_path(order, std::span<const Labels>{live_labels},
                                  plan.output_labels(), *bound);
  if (!chosen) {
    return std::unexpected{chosen.error()};
  }
  geom.path = *chosen;

  for (const auto &[si, step] : geom.path.steps | std::views::enumerate) {
    const Source &left = sources[step.l_src];
    const Source &right = sources[step.r_src];

    StepGeom sg;
    sg.l = make_slab(detail::pick(left.labels, left.layout, step.batch),
                     detail::pick(left.labels, left.layout, step.m),
                     detail::pick(left.labels, left.layout, step.k));
    sg.r = make_slab(detail::pick(right.labels, right.layout, step.batch),
                     detail::pick(right.labels, right.layout, step.k),
                     detail::pick(right.labels, right.layout, step.n));

    sg.l_scratch = left.scratch;
    sg.l_operand = left.operand;
    sg.l_offset = left.offset;
    sg.r_scratch = right.scratch;
    sg.r_operand = right.operand;
    sg.r_offset = right.offset;

    // The caller's output on the last step, otherwise a contiguous tensor in
    // batch ++ m ++ n order, which is trivially mappable.
    Labels target_labels = step.target;
    Layout target_layout;
    if (step.writes_output) {
      target_labels = plan.output_labels();
      target_layout = out;
      sg.out_scratch = false;
    } else {
      const Shape target_shape = shape_over(step.target, *bound);
      target_layout = make_layout(target_shape, row_major);
      sg.out_scratch = true;
      sg.out_offset = bump.take(product(target_shape));
    }
    sg.out = make_slab(detail::pick(target_labels, target_layout, step.batch),
                       detail::pick(target_labels, target_layout, step.m),
                       detail::pick(target_labels, target_layout, step.n));

    // Reused across batches, so one rectangle per unmappable side.
    if (!sg.l.mappable) {
      sg.l.pack_offset = bump.take(sg.m() * sg.k());
    }
    if (!sg.r.mappable) {
      sg.r.pack_offset = bump.take(sg.k() * sg.n());
    }
    if (!sg.out.mappable) {
      sg.out.pack_offset = bump.take(sg.m() * sg.n());
    }

    geom.steps.push_back(sg);

    sources[kIntermediate + static_cast<std::size_t>(si)] = {
        .labels = step.target,
        .layout = target_layout,
        .scratch = sg.out_scratch,
        .offset = sg.out_offset,
        .operand = 0};
  }

  geom.scratch_elems = bump.cursor;
  return geom;
}

} // namespace einsum::impl
