#pragma once

#include "einsum/core/lower.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/ranges.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <type_traits>

// The only place Eigen is called.  Strategy types rather than free functions:
// execute() selects one per step from the Geometry.
namespace einsum::impl {

// Inner stride 1 at compile time in both: Eigen's blas_traits evaluates an
// operand whose inner stride it cannot see is 1 into a heap temporary.  A
// column-major map keeps that for a transposed block, without a Transpose.
template <CScalar T>
using RowMap = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Unaligned, Eigen::OuterStride<>>;
template <CScalar T>
using CRowMap = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Unaligned, Eigen::OuterStride<>>;
template <CScalar T>
using ColMap = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
    Eigen::Unaligned, Eigen::OuterStride<>>;
template <CScalar T>
using CColMap = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
    Eigen::Unaligned, Eigen::OuterStride<>>;

// Eigen vectorises a vector Map whose inner stride is 1 *in the type*; one
// holding 1 in an InnerStride<> at run time it has to walk element by element.
// That is not a small difference -- summing 512 rows of 512 doubles measures
// 338us against 48us -- and every stride here is a runtime value, so the unit
// case has to be recovered rather than assumed.  Hence the stride parameter:
// the maps below are named with a stride *type*, and with_inner_stride picks
// which one, once, outside the loop that uses it.
template <CScalar T, typename Stride = Eigen::InnerStride<>>
using VecMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Unaligned, Stride>;
template <CScalar T, typename Stride = Eigen::InnerStride<>>
using CVecMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>,
                           Eigen::Unaligned, Stride>;

// A row vector, for the one product Eigen recognises only from the types.
template <CScalar T>
using CRowVecMap = Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>,
                              Eigen::Unaligned, Eigen::InnerStride<>>;

// A side that is packed by construction: contiguous, and known so at compile
// time, which is the whole point of packing it.
using Unit = Eigen::InnerStride<1>;

// Hands `f` an Eigen inner stride whose *type* says whether it is one.  Two
// instantiations of the body, and the branch is paid once rather than per
// element.
template <typename F>
constexpr void with_inner_stride(const index_t stride, F &&f) noexcept {
  if (stride == 1) {
    f(Unit{});
  } else {
    f(Eigen::InnerStride<>{stride});
  }
}

// The compile-time path's maps: every extent in the type, so the product is
// unrolled rather than blocked.  A single column has to be column-major and a
// single row row-major whatever the operand's order; Eigen refuses the rest.
[[nodiscard]] consteval int fixed_order(const index_t rows, const index_t cols,
                                        const bool row_order) noexcept {
  if (cols == 1) {
    return Eigen::ColMajor;
  }
  if (rows == 1) {
    return Eigen::RowMajor;
  }
  return row_order ? Eigen::RowMajor : Eigen::ColMajor;
}

template <CScalar T, index_t R, index_t C, bool RowOrder>
using CFixedMap =
    Eigen::Map<const Eigen::Matrix<T, static_cast<int>(R), static_cast<int>(C),
                                   fixed_order(R, C, RowOrder)>>;
template <CScalar T, index_t R, index_t C, bool RowOrder>
using FixedMap =
    Eigen::Map<Eigen::Matrix<T, static_cast<int>(R), static_cast<int>(C),
                             fixed_order(R, C, RowOrder)>>;

// A rectangle Eigen cannot address becomes one it can, and back again for an
// output.  One body, parameterised on which side is strided.
enum class Direction : std::uint8_t { gather, disperse };

// Constness follows the direction, so neither call site needs a cast.
template <Direction D, CScalar T>
using PackedPtr = std::conditional_t<D == Direction::gather, T *, const T *>;
template <Direction D, CScalar T>
using StridedPtr = std::conditional_t<D == Direction::gather, const T *, T *>;

template <Direction D, CScalar T>
void shuffle(PackedPtr<D, T> packed, StridedPtr<D, T> strided,
             const Layout &rows, const Layout &cols) noexcept {
  const Collapsed run = collapse(cols);
  const index_t cols_n = cols.size();
  // The packed side is contiguous by definition, so it is a Unit map and not a
  // dynamic 1; the strided side is dispatched on, and both -- along with the
  // run.ok branch -- are the same for every row.
  const auto lines = [&](auto &&body) noexcept {
    index_t r = 0;
    for_each_offset(
        [&](const index_t row_offset) noexcept {
          body(strided + row_offset, packed + r++ * cols_n);
        },
        rows);
  };
  if (run.ok) {
    with_inner_stride(run.stride, [&](const auto stride) noexcept {
      lines([&](auto *line, auto *flat) noexcept {
        if constexpr (D == Direction::gather) {
          VecMap<T, Unit>{flat, cols_n} =
              CVecMap<T, decltype(stride)>{line, cols_n, stride};
        } else {
          VecMap<T, decltype(stride)>{line, cols_n, stride} =
              CVecMap<T, Unit>{flat, cols_n};
        }
      });
    });
    return;
  }
  lines([&](auto *line, auto *flat) noexcept {
    index_t c = 0;
    for_each_offset(
        [&](const index_t col_offset) noexcept {
          if constexpr (D == Direction::gather) {
            flat[c] = line[col_offset];
          } else {
            line[col_offset] = flat[c];
          }
          ++c;
        },
        cols);
  });
}

// One GEMM per batch.  The nested dispatches pick each operand's Map order, so
// there is no runtime branch inside the product.
struct GemmKernel {
  template <CScalar T>
  static void run(const StepGeom &g, const T *left, const T *right, T *out,
                  T *scratch) noexcept {
    const index_t m = g.m();
    const index_t n = g.n();
    const index_t k = g.k();
    // The three batch groups name the same labels, so one walk serves all.
    for_each_offset(
        [&](const index_t lb, const index_t rb, const index_t ob) noexcept {
          const T *lp = left + lb;
          const T *rp = right + rb;
          T *op = out + ob;

          index_t l_outer = g.l.outer_stride;
          if (!g.l.mappable) {
            shuffle<Direction::gather, T>(scratch + g.l.pack_offset, lp,
                                          g.l.rows, g.l.cols);
            lp = scratch + g.l.pack_offset;
            l_outer = std::max<index_t>(k, 1);
          }
          index_t r_outer = g.r.outer_stride;
          if (!g.r.mappable) {
            shuffle<Direction::gather, T>(scratch + g.r.pack_offset, rp,
                                          g.r.rows, g.r.cols);
            rp = scratch + g.r.pack_offset;
            r_outer = std::max<index_t>(n, 1);
          }
          T *dp = op;
          index_t o_outer = g.out.outer_stride;
          if (!g.out.mappable) {
            dp = scratch + g.out.pack_offset;
            o_outer = std::max<index_t>(n, 1);
          }

          const bool l_col = g.l.mappable && g.l.transposed;
          const bool r_col = g.r.mappable && g.r.transposed;
          const bool o_col = g.out.mappable && g.out.transposed;

          const auto with_out = [&](const auto &a, const auto &bm) noexcept {
            if (o_col) {
              ColMap<T>{dp, m, n, Eigen::OuterStride<>{o_outer}}.noalias() =
                  a * bm;
            } else {
              RowMap<T>{dp, m, n, Eigen::OuterStride<>{o_outer}}.noalias() =
                  a * bm;
            }
          };
          const auto with_right = [&](const auto &a) noexcept {
            if (r_col) {
              with_out(a, CColMap<T>{rp, k, n, Eigen::OuterStride<>{r_outer}});
            } else {
              with_out(a, CRowMap<T>{rp, k, n, Eigen::OuterStride<>{r_outer}});
            }
          };
          // k == 1 is a rank-one update, and Eigen knows that only from the
          // types: an (m x 1) by (1 x n) product of two *matrix* maps is
          // packed and blocked like any GEMM -- 87us for 512 x 512 -- where
          // the same product of a column vector by a row vector takes its
          // outer-product path, one scaled row copy per row, 27us: the cost
          // of the write.  The vector's stride is whichever of the matrix's
          // two the single column or row runs along.
          if (k == 1) {
            with_out(CVecMap<T>{lp, m, Eigen::InnerStride<>{l_col ? 1 : l_outer}},
                     CRowVecMap<T>{rp, n,
                                   Eigen::InnerStride<>{r_col ? r_outer : 1}});
          } else if (l_col) {
            with_right(CColMap<T>{lp, m, k, Eigen::OuterStride<>{l_outer}});
          } else {
            with_right(CRowMap<T>{lp, m, k, Eigen::OuterStride<>{l_outer}});
          }

          if (!g.out.mappable) {
            shuffle<Direction::disperse, T>(dp, op, g.out.rows, g.out.cols);
          }
        },
        g.l.batch, g.r.batch, g.out.batch);
  }
};

// m == n == k == 1: every "matrix" is a scalar and the step is the batch.  One
// vector multiply when all three batch groups nest.
struct HadamardKernel {
  template <CScalar T>
  static void run(const StepGeom &g, const T *left, const T *right, T *out,
                  T * /*scratch*/) noexcept {
    const Collapsed lb = collapse(g.l.batch);
    const Collapsed rb = collapse(g.r.batch);
    const Collapsed ob = collapse(g.out.batch);
    const index_t batches = g.batches();
    if (lb.ok && rb.ok && ob.ok) {
      VecMap<T>{out, batches, Eigen::InnerStride<>{ob.stride}} =
          CVecMap<T>{left, batches, Eigen::InnerStride<>{lb.stride}}
              .cwiseProduct(
                  CVecMap<T>{right, batches, Eigen::InnerStride<>{rb.stride}});
      return;
    }
    for_each_offset(
        [&](const index_t lo, const index_t ro, const index_t oo) noexcept {
          out[oo] = left[lo] * right[ro];
        },
        g.l.batch, g.r.batch, g.out.batch);
  }
};

template <typename K, typename T>
concept CStepKernel =
    CScalar<T> && requires(const StepGeom &g, const T *in, T *out) {
      { K::template run<T>(g, in, in, out, out) } -> std::same_as<void>;
    };

static_assert(CStepKernel<GemmKernel, double>);
static_assert(CStepKernel<HadamardKernel, double>);

// Labels nothing else wants, summed before the operand enters a step, so the
// GEMM is over the smallest tensor that still answers the subscript.
struct ReduceKernel {
  template <CScalar T>
  static void run(const PrepGeom &pg, const T *src, T *dst) noexcept {
    // Whether there is a single stride, and what it is, are the same for every
    // row -- so both are settled out here, and the row loop is one sum with
    // nothing left to decide.
    if (pg.red_run.ok) {
      with_inner_stride(pg.red_run.stride, [&](const auto stride) noexcept {
        index_t i = 0;
        for_each_offset(
            [&](const index_t keep_offset) noexcept {
              dst[i++] = CVecMap<T, decltype(stride)>{
                  src + keep_offset, pg.red_run.extent, stride}
                             .sum();
            },
            pg.keep);
      });
      return;
    }
    // No single stride, so no vector for Eigen to sum.
    index_t i = 0;
    for_each_offset(
        [&](const index_t keep_offset) noexcept {
          const T *base = src + keep_offset;
          T acc{};
          for_each_offset([&](const index_t red) noexcept { acc += base[red]; },
                          pg.red);
          dst[i++] = acc;
        },
        pg.keep);
  }
};

// src is already permuted into the destination's axis order, so the two sides
// disagree only about strides and the innermost axis is one assignment.
struct PermuteKernel {
  template <CScalar T>
  static void run(const TensorView<T> &dst, const T *src,
                  const Layout &src_layout) noexcept {
    const std::size_t rank = dst.rank();
    if (rank == 0) {
      dst.data[0] = src[0];
      return;
    }
    const index_t inner = dst.layout.shape[rank - 1];
    const index_t dst_step = dst.layout.strides[rank - 1];
    const index_t src_step = src_layout.strides[rank - 1];

    // Everything but the innermost axis, which the vector assignment covers.
    const auto but_last = [rank](const Layout &lay) {
      return Layout{.shape = Shape{std::from_range,
                                   lay.shape | std::views::take(rank - 1)},
                    .strides = Shape{std::from_range,
                                     lay.strides | std::views::take(rank - 1)}};
    };
    // Both steps are the same for every line, and a transpose is exactly the
    // case where one of the two is 1: settled here rather than per element.
    with_inner_stride(dst_step, [&](const auto to_step) noexcept {
      with_inner_stride(src_step, [&](const auto from_step) noexcept {
        for_each_offset(
            [&](const index_t to, const index_t from) noexcept {
              VecMap<T, decltype(to_step)>{dst.data + to, inner, to_step} =
                  CVecMap<T, decltype(from_step)>{src + from, inner, from_step};
            },
            but_last(dst.layout), but_last(src_layout));
      });
    });
  }
};

} // namespace einsum::impl
