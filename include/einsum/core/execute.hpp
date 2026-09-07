#pragma once

#include "einsum/core/kernels.hpp"
#include "einsum/core/lower.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/concepts.hpp"

#include <ranges>
#include <span>

namespace einsum::impl {

// The whole evaluation: reduce, then either one permuted copy or a chain of
// GEMMs.  noexcept and allocation-free by construction -- every buffer it could
// want is an offset into `scratch`, sized by make_geometry -- and the only one
// of these there is: both front ends reach it with a TensorView and a span,
// whether the memory behind them is a std::array, a std::vector or the
// caller's own.
template <CScalar T>
void execute(const Plan &plan, const Geometry &geom,
             const std::span<const TensorView<const T>> views, const TensorView<T> &out,
             const std::span<T> scratch) noexcept {
  for (const auto &[view, pg] : std::views::zip(views, geom.preps)) {
    if (pg.reduced) {
      ReduceKernel::run<T>(pg, view.data, scratch.data() + pg.reduced_offset);
    }
  }

  if (plan.operand_count() == 1) {
    const T *src = geom.unary_from_scratch ? scratch.data() + geom.unary_offset : views[0].data;
    PermuteKernel::run<T>(out, src, geom.unary_src);
    return;
  }

  for (const StepGeom &sg : geom.steps) {
    const T *left = sg.l_scratch ? scratch.data() + sg.l_offset : views[sg.l_operand].data;
    const T *right = sg.r_scratch ? scratch.data() + sg.r_offset : views[sg.r_operand].data;
    T *dst = sg.out_scratch ? scratch.data() + sg.out_offset : out.data;
    // The Geometry chose the strategy; this is only where it is applied.
    if (sg.hadamard) {
      HadamardKernel::run<T>(sg, left, right, dst, scratch.data());
    } else {
      GemmKernel::run<T>(sg, left, right, dst, scratch.data());
    }
  }
}

} // namespace einsum::impl
