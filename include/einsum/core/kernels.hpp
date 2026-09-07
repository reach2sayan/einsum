#pragma once

#include "einsum/core/lower.hpp"
#include "einsum/core/odometer.hpp"
#include "einsum/util/concepts.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <ranges>
#include <cstdint>
#include <type_traits>

// The only place Eigen is called.  Everything above builds descriptions; each
// kernel below turns one into Map expressions, and every loop that could have
// been over elements is over a vector or a matrix instead.
//
// The kernels are strategy types rather than free functions: execute() selects
// one per step from the Geometry and knows nothing else about it.
namespace einsum::impl {

// Inner stride 1 at compile time in both, which is the whole point: Eigen's
// blas_traits refuses direct access to an operand whose inner stride it cannot
// see is 1, and evaluates it into a heap temporary instead.  A column-major map
// is how a transposed block keeps that property without a Transpose expression.
template <CScalar T>
using RowMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
                          Eigen::Unaligned, Eigen::OuterStride<>>;
template <CScalar T>
using CRowMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
                           Eigen::Unaligned, Eigen::OuterStride<>>;
template <CScalar T>
using ColMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                          Eigen::Unaligned, Eigen::OuterStride<>>;
template <CScalar T>
using CColMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                           Eigen::Unaligned, Eigen::OuterStride<>>;

template <CScalar T>
using VecMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Unaligned, Eigen::InnerStride<>>;
template <CScalar T>
using CVecMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Unaligned, Eigen::InnerStride<>>;

// The compile-time fast path's map: every extent in the type, so the product is
// unrolled rather than blocked.
template <CScalar T, index_t R, index_t C>
using CFixedMap =
    Eigen::Map<const Eigen::Matrix<T, static_cast<int>(R), static_cast<int>(C),
                                   (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>;
template <CScalar T, index_t R, index_t C>
using FixedMap = Eigen::Map<Eigen::Matrix<T, static_cast<int>(R), static_cast<int>(C),
                                          (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>;

// --- gathering ---------------------------------------------------------------
// A rectangle Eigen cannot address becomes one it can: rows x cols, row-major,
// contiguous.  scatter() is the same journey back, for an output whose strides
// Eigen could not write to.  One body, parameterised on which side is strided --
// the two differ only in that.
enum class Direction : std::uint8_t { gather, disperse };

// Constness follows the direction, so neither call site needs a cast; T is not
// deduced from either pointer and every caller names it.
template <Direction D, CScalar T>
using PackedPtr = std::conditional_t<D == Direction::gather, T *, const T *>;
template <Direction D, CScalar T>
using StridedPtr = std::conditional_t<D == Direction::gather, const T *, T *>;

template <Direction D, CScalar T>
void shuffle(PackedPtr<D, T> packed, StridedPtr<D, T> strided, const Layout &rows,
             const Layout &cols, const index_t rows_n, const index_t cols_n) noexcept {
  const Collapsed run = collapse(cols);
  Odometer row{rows};
  for (index_t r = 0; r < rows_n; ++r, ++row) {
    auto *line = strided + row.offset();
    auto *flat = packed + r * cols_n;
    if (run.ok) {
      if constexpr (D == Direction::gather) {
        VecMap<T>{flat, cols_n, Eigen::InnerStride<>{1}} =
            CVecMap<T>{line, cols_n, Eigen::InnerStride<>{run.stride}};
      } else {
        VecMap<T>{line, cols_n, Eigen::InnerStride<>{run.stride}} =
            CVecMap<T>{flat, cols_n, Eigen::InnerStride<>{1}};
      }
    } else {
      Odometer col{cols};
      for (index_t c = 0; c < cols_n; ++c, ++col) {
        if constexpr (D == Direction::gather) {
          flat[c] = line[col.offset()];
        } else {
          line[col.offset()] = flat[c];
        }
      }
    }
  }
}

// --- the step strategies -----------------------------------------------------
// One GEMM per batch.  The three nested dispatches pick a row- or column-major
// Map per operand; there is no runtime branch inside the product itself.
struct GemmKernel {
  template <CScalar T>
  static void run(const StepGeom &g, const T *left, const T *right, T *out,
                  T *scratch) noexcept {
    Odometer lb{g.l.batch};
    Odometer rb{g.r.batch};
    Odometer ob{g.out.batch};
    for (index_t b = 0; b < g.batches; ++b, ++lb, ++rb, ++ob) {
      const T *lp = left + lb.offset();
      const T *rp = right + rb.offset();
      T *op = out + ob.offset();

      index_t l_outer = g.l.outer_stride;
      if (!g.l.mappable) {
        shuffle<Direction::gather, T>(scratch + g.l.pack_offset, lp, g.l.rows, g.l.cols, g.m,
                                      g.k);
        lp = scratch + g.l.pack_offset;
        l_outer = std::max<index_t>(g.k, 1);
      }
      index_t r_outer = g.r.outer_stride;
      if (!g.r.mappable) {
        shuffle<Direction::gather, T>(scratch + g.r.pack_offset, rp, g.r.rows, g.r.cols, g.k,
                                      g.n);
        rp = scratch + g.r.pack_offset;
        r_outer = std::max<index_t>(g.n, 1);
      }
      T *dp = op;
      index_t o_outer = g.out.outer_stride;
      if (!g.out.mappable) {
        dp = scratch + g.out.pack_offset;
        o_outer = std::max<index_t>(g.n, 1);
      }

      const bool l_col = g.l.mappable && g.l.transposed;
      const bool r_col = g.r.mappable && g.r.transposed;
      const bool o_col = g.out.mappable && g.out.transposed;

      const auto with_out = [&](const auto &a, const auto &bm) noexcept {
        if (o_col) {
          ColMap<T>{dp, g.m, g.n, Eigen::OuterStride<>{o_outer}}.noalias() = a * bm;
        } else {
          RowMap<T>{dp, g.m, g.n, Eigen::OuterStride<>{o_outer}}.noalias() = a * bm;
        }
      };
      const auto with_right = [&](const auto &a) noexcept {
        if (r_col) {
          with_out(a, CColMap<T>{rp, g.k, g.n, Eigen::OuterStride<>{r_outer}});
        } else {
          with_out(a, CRowMap<T>{rp, g.k, g.n, Eigen::OuterStride<>{r_outer}});
        }
      };
      if (l_col) {
        with_right(CColMap<T>{lp, g.m, g.k, Eigen::OuterStride<>{l_outer}});
      } else {
        with_right(CRowMap<T>{lp, g.m, g.k, Eigen::OuterStride<>{l_outer}});
      }

      if (!g.out.mappable) {
        shuffle<Direction::disperse, T>(dp, op, g.out.rows, g.out.cols, g.m, g.n);
      }
    }
  }
};

// m == n == k == 1: every "matrix" is a scalar and the whole step is the batch.
// One vector multiply when all three batch groups nest, which they do whenever
// the operands agree on their axis order.
struct HadamardKernel {
  template <CScalar T>
  static void run(const StepGeom &g, const T *left, const T *right, T *out,
                  T * /*scratch*/) noexcept {
    const Collapsed lb = collapse(g.l.batch);
    const Collapsed rb = collapse(g.r.batch);
    const Collapsed ob = collapse(g.out.batch);
    if (lb.ok && rb.ok && ob.ok) {
      VecMap<T>{out, g.batches, Eigen::InnerStride<>{ob.stride}} =
          CVecMap<T>{left, g.batches, Eigen::InnerStride<>{lb.stride}}.cwiseProduct(
              CVecMap<T>{right, g.batches, Eigen::InnerStride<>{rb.stride}});
      return;
    }
    Odometer lo{g.l.batch};
    Odometer ro{g.r.batch};
    Odometer oo{g.out.batch};
    for (index_t b = 0; b < g.batches; ++b, ++lo, ++ro, ++oo) {
      out[oo.offset()] = left[lo.offset()] * right[ro.offset()];
    }
  }
};

template <typename K, typename T>
concept CStepKernel = CScalar<T> && requires(const StepGeom &g, const T *in, T *out) {
  { K::template run<T>(g, in, in, out, out) } -> std::same_as<void>;
};

static_assert(CStepKernel<GemmKernel, double>);
static_assert(CStepKernel<HadamardKernel, double>);

// --- the sum a contraction never sees ----------------------------------------
// Labels no other operand has and the output does not want, summed before the
// operand enters a step -- so the GEMM above is over the smallest tensor that
// still answers the subscript.
struct ReduceKernel {
  template <CScalar T>
  static void run(const PrepGeom &pg, const T *src, T *dst) noexcept {
    const index_t keep_n = pg.keep.size();
    const index_t red_n = pg.red.size();
    Odometer keep{pg.keep};
    for (index_t i = 0; i < keep_n; ++i, ++keep) {
      const T *base = src + keep.offset();
      if (pg.red_run.ok) {
        dst[i] = CVecMap<T>{base, pg.red_run.extent, Eigen::InnerStride<>{pg.red_run.stride}}
                     .sum();
      } else {
        T acc{};
        Odometer red{pg.red};
        for (index_t j = 0; j < red_n; ++j, ++red) {
          acc += base[red.offset()];
        }
        dst[i] = acc;
      }
    }
  }
};

// --- the one-operand strategy ------------------------------------------------
// src is already permuted into the destination's axis order, so this is a copy
// whose two sides disagree only about strides.  The innermost axis is one
// vector assignment, whatever either side's stride along it happens to be.
struct PermuteKernel {
  template <CScalar T>
  static void run(const TensorView<T> &dst, const T *src, const Layout &src_layout) noexcept {
    const std::size_t rank = dst.rank();
    if (rank == 0) {
      dst.data[0] = src[0];
      return;
    }
    const index_t inner = dst.layout.shape[rank - 1];
    const index_t dst_step = dst.layout.strides[rank - 1];
    const index_t src_step = src_layout.strides[rank - 1];

    // Everything but the innermost axis, which the vector assignment covers.
    const auto but_last = [rank](const Shape &s) {
      return Shape{s | std::views::take(rank - 1)};
    };
    const Shape outer = but_last(dst.layout.shape);
    Odometer to{outer, but_last(dst.layout.strides)};
    Odometer from{outer, but_last(src_layout.strides)};
    const index_t rows = product(outer);
    for (index_t o = 0; o < rows; ++o, ++to, ++from) {
      VecMap<T>{dst.data + to.offset(), inner, Eigen::InnerStride<>{dst_step}} =
          CVecMap<T>{src + from.offset(), inner, Eigen::InnerStride<>{src_step}};
    }
  }
};

} // namespace einsum::impl
