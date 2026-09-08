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

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <span>

// The one call whose arity and ranks are values rather than types -- what a
// language binding reading shapes off an array object needs, since it cannot
// reach BasicEinsum::operator(), whose every rank is in a type.  Everything
// below the operand adaptation is already dynamic, so this is evaluate() with
// the templates taken out, sharing the object's own lowering cache with it.
namespace einsum::impl {

// allocate(shape) hands back a contiguous row-major buffer of that shape, the
// caller's own, called once the lowering says what shape the result is.  The
// rank is the subscript's rather than any operand's, which is what makes
// "i,j->ij" reachable here.
template <typename A, typename T>
concept CAllocates = requires(A &&allocate, const Shape &shape) {
  { allocate(shape) } -> std::same_as<T *>;
};

template <CScalar T>
[[nodiscard]] result<void>
evaluate_dynamic(const Einsum &e,
                 const std::span<const TensorView<const T>> ops,
                 CAllocates<T> auto &&allocate) {
  if (ops.size() != e.operand_count()) {
    return fail(errc::operand_count_mismatch);
  }
  if (ops.size() > kMaxOperands) {
    return fail(errc::too_many_operands);
  }

  if (std::ranges::any_of(ops, [](const TensorView<const T> &view) {
        return view.rank() > kMaxRank;
      })) {
    return fail(errc::rank_too_high);
  }
  Layouts lays;
  std::ranges::transform(ops, std::back_inserter(lays),
                         [](const TensorView<const T> &view) -> const Layout & {
                           return view.layout;
                         });
  const std::span<const Layout> spans{lays};

  // The same cache the typed call operator keeps, keyed the same way on the
  // operand layouts.  The kind is what keeps the two apart: a TensorView is
  // never a result type, so &result_id<TensorView<T>> is an address no typed
  // call can produce.
  if (const auto ok = refresh(einsum_access::lowering_of(e), einsum_access::plan_of(e),
                              einsum_access::order_of(e), spans,
                              &result_id<TensorView<T>>,
                              [](const Shape &shape) {
                                return make_layout<RowMajor>(shape);
                              });
      !ok) {
    return propagate<void>(ok.error());
  }
  const LastLowering &lowering = einsum_access::lowering_of(e);

  T *const out_data = allocate(lowering.out_shape);

  // No scratch_offsets here: every operand is a strided rectangle the kernels
  // address where it lies and the output is contiguous, so the block is the
  // geometry's own working space and nothing else.
  const auto elems = static_cast<std::size_t>(lowering.geometry.scratch_elems);
  const std::span block =
      einsum_access::scratch_of(e).bytes(elems * sizeof(T));
  T *const pool = block.empty() ? nullptr : reinterpret_cast<T *>(block.data());

  execute(lowering.plan, lowering.geometry, ops,
          TensorView{out_data, lowering.out_layout}, std::span{pool, elems});
  return {};
}

} // namespace einsum::impl
