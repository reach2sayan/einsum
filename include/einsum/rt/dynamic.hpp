#pragma once

#include "einsum/core/einsum_object.hpp"
#include "einsum/core/execute.hpp"
#include "einsum/core/limits.hpp"
#include "einsum/core/lower.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/core/view.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"

#include <concepts>
#include <cstddef>
#include <span>

// The one call whose arity and ranks are values rather than types.
//
// BasicEinsum::operator() is a variadic template over COperand, so every
// operand's rank is in its type and every call's arity is in its signature.
// With kMaxRank and kMaxOperands both 8, a caller that learns all of that at
// run time -- a language binding, reading shapes off an array object -- cannot
// reach it: the cross-product is not a table anybody builds.
//
// It does not have to.  Everything below the operand adaptation is already
// dynamic: an impl::Layout is extents and strides and no type, and expand,
// make_plan, infer_output_shape, make_geometry and execute all take spans.  So
// this is evaluate() with the templates taken out, sharing the object's own
// lowering cache with it, and templated on the scalar alone.
//
// Not part of the public surface: einsum::impl, and everything it takes is one.
namespace einsum::impl {

// allocate(shape) hands back a row-major buffer of that shape, contiguous and
// owned by the caller.  It is called once, after the lowering says what shape
// the result is, and its return is where the contraction is written -- so the
// caller allocates the exact rank the subscript implies and nothing pads or
// fits it afterwards.  That is what makes "i,j->ij" reachable here: the rank
// comes from the subscript, not from an operand's type.
template <typename A, typename T>
concept CAllocates = requires(A &&allocate, const Shape &shape) {
  { allocate(shape) } -> std::same_as<T *>;
};

template <CScalar T, CAllocates<T> Alloc>
[[nodiscard]] result<void>
evaluate_dynamic(const Einsum &e, const std::span<const TensorView<const T>> ops,
                 Alloc &&allocate) {
  if (ops.size() != e.operand_count()) {
    return fail(errc::operand_count_mismatch);
  }
  if (ops.size() > kMaxOperands) {
    return fail(errc::too_many_operands);
  }

  Layouts lays;
  for (const TensorView<const T> &view : ops) {
    if (view.rank() > kMaxRank) {
      return fail(errc::rank_too_high);
    }
    lays.push_back(view.layout);
  }
  const std::span<const Layout> spans{lays};

  // The same cache the typed call operator keeps, keyed the same way on the
  // operand layouts.  The kind is what keeps the two apart: a TensorView is
  // never a result type, so &result_id<TensorView<T>> is an address no typed
  // call can produce and a dynamic call cannot collide with one.
  LastLowering &lowering = einsum_access::lowering_of(e);
  const void *const kind = &result_id<TensorView<T>>;
  if (!lowering.matches(spans, kind)) {
    FixedVec<std::uint8_t, kMaxOperands> ranks;
    for (const Layout &lay : lays) {
      ranks.push_back(static_cast<std::uint8_t>(lay.rank()));
    }
    const auto expanded =
        expand(e.subscripts(), std::span<const std::uint8_t>{ranks});
    if (!expanded) {
      return propagate<void>(expanded.error());
    }
    const auto fresh_plan = make_plan(*expanded);
    if (!fresh_plan) {
      return propagate<void>(fresh_plan.error());
    }
    const auto fresh_shape = infer_output_shape(*fresh_plan, spans);
    if (!fresh_shape) {
      return propagate<void>(fresh_shape.error());
    }
    const Layout fresh_layout = make_layout<RowMajor>(*fresh_shape);
    const auto fresh_geometry = make_geometry(*fresh_plan, spans, fresh_layout,
                                              einsum_access::order_of(e));
    if (!fresh_geometry) {
      return propagate<void>(fresh_geometry.error());
    }
    lowering.operands.assign(lays.begin(), lays.end());
    lowering.plan = *fresh_plan;
    lowering.out_shape = *fresh_shape;
    lowering.out_layout = fresh_layout;
    lowering.geometry = *fresh_geometry;
    lowering.result_kind = kind;
  }

  T *const out_data = allocate(lowering.out_shape);

  // No scratch_offsets here, and nothing for it to lay out: every operand is a
  // strided rectangle the kernels address where it lies, so none is packed, and
  // the output is contiguous and ours, so nothing is scattered afterwards.  The
  // block is the geometry's own working space and that alone.
  const auto elems = static_cast<std::size_t>(lowering.geometry.scratch_elems);
  const std::span<std::byte> block =
      einsum_access::scratch_of(e).bytes(elems * sizeof(T));
  T *const pool = block.empty() ? nullptr : reinterpret_cast<T *>(block.data());

  execute<T>(lowering.plan, lowering.geometry, ops,
             TensorView<T>{out_data, lowering.out_layout},
             std::span<T>{pool, elems});
  return {};
}

} // namespace einsum::impl
