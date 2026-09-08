#pragma once

#include "einsum/core/execute.hpp"
#include "einsum/core/kind.hpp"
#include "einsum/core/lower.hpp"
#include "einsum/core/path.hpp"
#include "einsum/core/owned.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <boost/container/static_vector.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsum {

namespace impl {

template <typename S>
concept CScratch = requires(const S &s, std::size_t n) {
  { s.bytes(n) } -> std::same_as<std::span<std::byte>>;
};

class HeapScratch {
public:
  [[nodiscard]] std::span<std::byte> bytes(const std::size_t n) const {
    if (buffer_.size() < n) {
      buffer_.resize(n);
    }
    return std::span{buffer_}.first(n);
  }

private:
  mutable std::vector<std::byte> buffer_;
};

template <typename R> inline constexpr char result_id = 0;

struct LastLowering {
  boost::container::static_vector<Layout, kMaxOperands> operands{};
  // Expanded against the last operands' ranks, when the subscript had '...'.
  Plan plan{};
  Shape out_shape{};
  Layout out_layout{};
  Geometry geometry{};
  const void *result_kind = nullptr;

  [[nodiscard]] bool matches(const std::span<const Layout> lays,
                             const void *kind) const noexcept {
    return result_kind == kind && std::ranges::equal(operands, lays);
  }
};

// The cached lowering, brought up to date for these layouts.  out_layout_of is
// where the two entry points differ.
template <typename LayoutFn>
  requires std::invocable<LayoutFn &, const Shape &>
[[nodiscard]] result<void> refresh(LastLowering &lowering, const Plan &plan,
                                   const path order,
                                   const std::span<const Layout> lays,
                                   const void *kind,
                                   LayoutFn &&out_layout_of) {
  if (lowering.matches(lays, kind)) {
    return {};
  }
  const auto ranks = ranks_of(lays);
  const auto expanded =
      expand(plan.subscripts(), std::span<const std::uint8_t>{ranks});
  if (!expanded) {
    return std::unexpected{expanded.error()};
  }
  const auto fresh_plan = make_plan(*expanded);
  if (!fresh_plan) {
    return std::unexpected{fresh_plan.error()};
  }
  const auto fresh_shape = infer_output_shape(*fresh_plan, lays);
  if (!fresh_shape) {
    return std::unexpected{fresh_shape.error()};
  }
  const Layout fresh_layout = out_layout_of(*fresh_shape);
  const auto fresh_geometry =
      make_geometry(*fresh_plan, lays, fresh_layout, order);
  if (!fresh_geometry) {
    return std::unexpected{fresh_geometry.error()};
  }
  lowering.operands.assign(lays.begin(), lays.end());
  lowering.plan = *fresh_plan;
  lowering.out_shape = *fresh_shape;
  lowering.out_layout = fresh_layout;
  lowering.geometry = *fresh_geometry;
  lowering.result_kind = kind;
  return {};
}

using Layouts = boost::container::static_vector<Layout, kMaxOperands>;

// Its own strides when the kernels can address it, row-major when it is packed.
template <COperand X>
[[nodiscard]] constexpr Layout layout_for(const X &x,
                                          const Shape &shape) noexcept {
  if constexpr (CContiguous<X>) {
    if constexpr (CEigenTensor<std::remove_cvref_t<X>>) {
      return tensor_layout<std::remove_cvref_t<X>>(shape);
    } else if constexpr (CEigenDense<std::remove_cvref_t<X>>) {
      return eigen_layout(x);
    } else {
      const auto axes = std::views::iota(std::size_t{0}, rank_v<X>);
      return {.shape = {std::from_range,
                        axes | std::views::transform([&](const std::size_t i) {
                          return static_cast<index_t>(x.extent(i));
                        })},
              .strides = {std::from_range,
                          axes |
                              std::views::transform([&](const std::size_t i) {
                                return static_cast<index_t>(x.stride(i));
                              })}};
    }
  } else {
    return make_layout<RowMajor>(shape);
  }
}

// At the rank the subscript produced, not the result type's own: an Eigen
// matrix holding a rank-1 result is Nx1, and the executor sees one axis.
template <COperand R>
[[nodiscard]] constexpr Layout layout_of_result(const Shape &shape) noexcept {
  if constexpr (CEigenTensor<R>) {
    return tensor_layout<R>(shape);
  } else if constexpr (CMdarray<R>) {
    return make_layout<RowMajor>(shape); // layout_right
  } else if constexpr (CEigenDense<R>) {
    Layout laid{.shape = shape, .strides = {}};
    if (shape.size() == 1) {
      laid.strides.push_back(1);
    } else if (shape.size() == 2) {
      laid.strides.push_back(R::IsRowMajor ? shape[1] : 1);
      laid.strides.push_back(R::IsRowMajor ? 1 : shape[0]);
    }
    return laid;
  } else {
    return make_layout<RowMajor>(shape);
  }
}

template <COperand X>
[[nodiscard]] constexpr const scalar_of_t<X> *data_of(const X &x) noexcept {
  if constexpr (CEigenDense<std::remove_cvref_t<X>>) {
    return x.data();
  } else {
    return x.data_handle();
  }
}

// Enough of a tuple for tuple_size, tuple_element_t and get.
template <typename Tup>
concept CTupleLike = requires { std::tuple_size<std::remove_cvref_t<Tup>>::value; };

// Is the last argument an output?  Neither the operand count nor constness can
// answer that at run time, so out(x) is the whole disambiguator.
template <typename X> inline constexpr bool is_out_tag_v = false;

template <typename... A>
[[nodiscard]] consteval bool is_rt_out_form() noexcept {
  if constexpr (sizeof...(A) < 2) {
    // One argument is an operand: a subscript naming none does not parse.
    return false;
  } else {
    return is_out_tag_v<std::remove_cvref_t<
        std::tuple_element_t<sizeof...(A) - 1, std::tuple<A...>>>>;
  }
}

// The one scratch block: the geometry's workspace first, since execute()
// addresses it from zero, then the packed operands, then the output when it
// cannot be written into directly.
struct ScratchMap {
  std::array<index_t, kMaxOperands> packed_at{}; // -1 where nothing is packed
  index_t out_at = -1;                           // -1 when the output is direct
  index_t total = 0;
};

[[nodiscard]] constexpr ScratchMap
scratch_offsets(const std::span<const Layout> lays,
                const std::span<const bool> gathered, const index_t scratch_elems,
                const Shape &out_shape, const bool out_direct) noexcept {
  ScratchMap map;
  map.packed_at.fill(-1);
  index_t cursor = scratch_elems;
  for (const auto &[at, packs, lay] :
       std::views::zip(map.packed_at, gathered, lays)) {
    if (packs) {
      at = cursor;
      cursor += lay.size();
    }
  }
  if (!out_direct) {
    map.out_at = cursor;
    cursor += product(out_shape);
  }
  map.total = cursor;
  return map;
}

// `pool` is the block scratch_offsets() described.
template <CScalar T, COperand R, COperand... Ops>
void contract_into(const Plan &plan, const Geometry &geometry,
                   const std::span<const Layout> lays, const Layout &out_layout,
                   T *const pool, const ScratchMap &map, R &out,
                   const Shape &fitted, const Ops &...ops) {
  boost::container::static_vector<TensorView<const T>, kMaxOperands> views;
  std::size_t next = 0;
  const auto prepare = [&](const auto &op) {
    const T *base = nullptr;
    if constexpr (CContiguous<decltype(op)>) {
      base = data_of(op);
    } else {
      T *const packed = pool + map.packed_at[next];
      gather(op, packed, lays[next].shape);
      base = packed;
    }
    views.push_back(TensorView<const T>{base, lays[next]});
    ++next;
  };
  (prepare(ops), ...);

  // if constexpr, not a ternary: the two branches have different pointer types.
  T *target_data = nullptr;
  if constexpr (CDirectWritable<R>) {
    target_data = out.data();
  } else {
    target_data = pool + map.out_at;
  }
  execute<T>(plan, geometry, std::span<const TensorView<const T>>{views},
             TensorView<T>{target_data, out_layout},
             std::span<T>{pool, static_cast<std::size_t>(geometry.scratch_elems)});

  if constexpr (!CDirectWritable<R>) {
    scatter(static_cast<const T *>(pool + map.out_at), out, fitted);
  }
}


} // namespace impl

// Marks the last argument as the result, which is what lets a call reach a rank
// no operand implies: "ij,kl->ijkl" from two rank-2 operands.
template <COperand R> struct Out {
  R &target;
};

template <COperand R> [[nodiscard]] constexpr Out<R> out(R &target) noexcept {
  return {target};
}

namespace impl {
template <typename R> inline constexpr bool is_out_tag_v<Out<R>> = true;
} // namespace impl

// A parsed subscript and nothing else a caller can see.  Both call operators
// are const, but concurrent calls on one object share its caches and so need
// synchronising.
template <impl::CScratch Scratch> class BasicEinsum;

namespace impl {
// The one way an Einsum is built from a Plan, which a caller never has.
struct einsum_access;
} // namespace impl

template <impl::CScratch Scratch> class BasicEinsum {
public:
  // A copy starts cold: the caches belong to the object that warmed them.
  BasicEinsum(const BasicEinsum &other) : plan_{other.plan_}, order_{other.order_} {}
  BasicEinsum &operator=(const BasicEinsum &other) {
    plan_ = other.plan_;
    order_ = other.order_;
    scratch_ = Scratch{};
    lowering_ = impl::LastLowering{};
    return *this;
  }
  BasicEinsum(BasicEinsum &&) noexcept = default;
  BasicEinsum &operator=(BasicEinsum &&) noexcept = default;
  ~BasicEinsum() = default;

  [[nodiscard]] constexpr const Subscripts &subscripts() const noexcept {
    return plan_.subscripts();
  }
  [[nodiscard]] constexpr std::size_t operand_count() const noexcept {
    return plan_.operand_count();
  }
  [[nodiscard]] constexpr const Labels &output_labels() const noexcept {
    return plan_.output_labels();
  }

  // obj(a, b, ...)         -> result<R>, in the operands' own family
  // obj(a, b, ..., out(x)) -> result<void>, moved into x
  template <typename... A> [[nodiscard]] auto operator()(A &&...args) const {
    if constexpr (impl::is_rt_out_form<A...>()) {
      return untag_output(std::forward_as_tuple(EINSUM_FWD(args)...),
                          std::make_index_sequence<sizeof...(A) - 1>{});
    } else {
      static_assert(CSameFamily<A...>,
                    "einsum(...): one call's operands must be one family and "
                    "one scalar -- all Eigen objects, all mdspans or spans, or "
                    "all nested ranges; their ranks may differ.  A trailing "
                    "non-const operand is read as an output, so pass operands "
                    "as const");
      if constexpr (!CSameFamily<A...>) {
        return result<void>{};
      } else {
        return by_value(args...);
      }
    }
  }

protected:
  BasicEinsum() = default;
  constexpr BasicEinsum(Plan plan, const path order) noexcept
      : plan_{std::move(plan)}, order_{order} {}
  friend struct impl::einsum_access;

  // Drop the tag; both objects come through here, so out() means one thing.
  template <impl::CTupleLike Tup, std::size_t... I>
  result<void> untag_output(Tup tup, std::index_sequence<I...>) const {
    return with_output(std::forward_as_tuple(
                           std::get<I>(tup)...,
                           std::get<sizeof...(I)>(tup).target),
                       std::index_sequence<I...>{});
  }

  // Bare types, not the forwarding references above: result_of_t is written in
  // terms of operand types, and a deduced `T&` is not one.
  template <COperand... Ops>
    requires CSameFamily<Ops...>
  [[nodiscard]] result<result_of_t<Ops...>> by_value(const Ops &...ops) const {
    return evaluate<result_of_t<Ops...>, scalar_of_t<impl::first_of_t<Ops...>>>(
        ops...);
  }

  // R is the output's own type, not result_of_t of the operands, so this form
  // reaches ranks by_value cannot.
  template <impl::CTupleLike Tup, std::size_t... I>
  result<void> with_output(Tup tup, std::index_sequence<I...>) const {
    auto &out = std::get<sizeof...(I)>(tup);
    using R = std::remove_cvref_t<decltype(out)>;
    return evaluate<R, scalar_of_t<
                           impl::first_of_t<std::tuple_element_t<I, Tup>...>>>(
               std::get<I>(tup)...)
        .transform([&out](R &&value) noexcept { out = std::move(value); });
  }

  template <COperand R, CScalar T, COperand... Ops>
  [[nodiscard]] result<R> evaluate(const Ops &...ops) const;

  Plan plan_{};
  // Part of what the object is, not of what it caches.
  path order_ = path::greedy;
  Scratch scratch_{};
  // Per-object and not copied.
  mutable impl::LastLowering lowering_{};
};

// Every failure is answered before an element moves.
template <impl::CScratch Scratch>
template <COperand R, CScalar T, COperand... Ops>
result<R> BasicEinsum<Scratch>::evaluate(const Ops &...ops) const {
  constexpr std::size_t kOperands = sizeof...(Ops);
  if (kOperands != plan_.operand_count()) {
    return fail(errc::operand_count_mismatch);
  }

  const std::array<result<Shape>, kOperands> shapes{impl::shape_of(ops)...};
  if (const auto bad = std::ranges::find_if(
          shapes, [](const result<Shape> &s) { return !s.has_value(); });
      bad != shapes.end()) {
    return propagate<R>(bad->error());
  }

  impl::Layouts lays;
  std::size_t next = 0;
  (lays.push_back(impl::layout_for(ops, *shapes[next++])), ...);
  const std::span<const impl::Layout> spans{lays};

  // Keyed on the operand layouts alone.  Only the description is cached; the
  // contraction below always runs.
  constexpr bool kOutDirect = CDirectWritable<R>;
  const void *const kind = &impl::result_id<R>;
  if (const auto ok = impl::refresh(
          lowering_, plan_, order_, spans, kind,
          [](const Shape &shape) { return impl::layout_of_result<R>(shape); });
      !ok) {
    return propagate<R>(ok.error());
  }
  const Plan &plan = lowering_.plan;
  const Shape &out_shape = lowering_.out_shape;
  const impl::Layout &out_layout = lowering_.out_layout;
  const impl::Geometry &geometry = lowering_.geometry;

  auto out = impl::make_like<R>(out_shape);
  if (!out) {
    return propagate<R>(out.error());
  }
  // The result's own rank; same elements in the same order, so the executor is
  // unaffected.
  const auto fitted = impl::fit_shape(
      out_shape, impl::CEigenDense<R> ? std::size_t{2} : rank_v<R>);
  if (!fitted) {
    return propagate<R>(fitted.error());
  }

  // A pooled allocation here, an array in the frame on the compile-time path.
  constexpr std::array<bool, kOperands> kGathered{!impl::CContiguous<Ops>...};
  const impl::ScratchMap map = impl::scratch_offsets(
      spans, std::span<const bool>{kGathered}, geometry.scratch_elems, out_shape,
      kOutDirect);

  const std::span<std::byte> block =
      scratch_.bytes(static_cast<std::size_t>(map.total) * sizeof(T));
  T *const pool = block.empty() ? nullptr : reinterpret_cast<T *>(block.data());

  impl::contract_into<T>(plan, geometry, spans, out_layout, pool, map, *out,
                         *fitted, ops...);
  return std::move(*out);
}

namespace impl {
struct einsum_access {
  template <CScratch Scratch>
  [[nodiscard]] static BasicEinsum<Scratch> make(Plan plan, const path order) noexcept {
    return BasicEinsum<Scratch>{std::move(plan), order};
  }

  // For tests; neither is on the public surface.
  template <CScratch Scratch>
  [[nodiscard]] static path order_of(const BasicEinsum<Scratch> &e) noexcept {
    return e.order_;
  }
  template <CScratch Scratch>
  [[nodiscard]] static const Path &last_path(const BasicEinsum<Scratch> &e) noexcept {
    return e.lowering_.geometry.path;
  }

  // What <einsum/rt/dynamic.hpp> needs and a caller does not.
  template <CScratch Scratch>
  [[nodiscard]] static const Plan &
  plan_of(const BasicEinsum<Scratch> &e) noexcept {
    return e.plan_;
  }
  template <CScratch Scratch>
  [[nodiscard]] static LastLowering &
  lowering_of(const BasicEinsum<Scratch> &e) noexcept {
    return e.lowering_;
  }
  template <CScratch Scratch>
  [[nodiscard]] static const Scratch &
  scratch_of(const BasicEinsum<Scratch> &e) noexcept {
    return e.scratch_;
  }

  // A constant, so a test asserts it rather than running it.
  template <std::derived_from<BasicEinsum<HeapScratch>> E, COperand... Ops>
  [[nodiscard]] static consteval Path static_path() noexcept {
    return E::template Lowered<Ops...>::geometry.path;
  }
};
} // namespace impl

using Einsum = BasicEinsum<impl::HeapScratch>;

} // namespace einsum
