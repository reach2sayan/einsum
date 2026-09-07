#pragma once

#include "einsum/core/bound.hpp"
#include "einsum/core/execute.hpp"
#include "einsum/core/kind.hpp"

#include "einsum/core/lower.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/core/owned.hpp"
#include "einsum/core/view.hpp"
#include "einsum/ct/labels.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/fixed_string.hpp"

#include <Eigen/Core>
#include <experimental/mdspan>

#include <algorithm>
#include <array>
#include <span>
#include <type_traits>

namespace einsum {

// The compile-time entry point.  Same Subscripts, same Plan, same Geometry and
// the same kernels as the runtime path -- everything the runtime path decides
// while running, this one has already decided, and what is left is a result
// array, a scratch array and one call.
//
// Every way the subscript or the operands can be wrong is a static_assert
// carrying the sentence the runtime error would have carried.
template <impl::FixedString S, ct::CStaticOperand... Ops> class CtEinsum {
  static constexpr std::size_t kOperands = sizeof...(Ops);
  static_assert(kOperands > 0, "einsum<\"...\">: needs at least one operand");
  static_assert(CHomogeneous<Ops...>,
                "einsum<\"...\">: every operand must be the same kind -- all Eigen "
                "objects or all mdspans -- and all the same scalar type");

public:
  using value_type = common_value_t<Ops...>;

  // The rung the operands stand on, which is also the rung the result comes
  // back on.  Only two of them can be a compile-time operand at all: a range
  // and a pointer have no shape in their type.
  static constexpr OperandKind kind = common_kind_v<Ops...>;

private:
  static constexpr Subscripts kSubscripts = ct::subscripts<S>::value;
  static_assert(kSubscripts.operands.size() == kOperands,
                "einsum<\"...\">: the subscript names a different number of operands "
                "than the call passes");

  static constexpr auto kPlanResult = impl::make_plan(kSubscripts);
#define EINSUM_PLAN_FAILED(code) failed_with(kPlanResult, errc::code)
  EINSUM_ASSERT_NO_ERROR(EINSUM_PLAN_FAILED)
#undef EINSUM_PLAN_FAILED
  static constexpr Plan kPlan = kPlanResult.value_or(Plan{});

  static constexpr std::array<Layout, kOperands> kLayouts{
      ct::operand_traits<std::remove_cvref_t<Ops>>::layout...};

  static constexpr auto kShapeResult =
      impl::infer_output_shape(kPlan, std::span<const Layout>{kLayouts});
#define EINSUM_SHAPE_FAILED(code) failed_with(kShapeResult, errc::code)
  EINSUM_ASSERT_NO_ERROR(EINSUM_SHAPE_FAILED)
#undef EINSUM_SHAPE_FAILED
  static constexpr Shape kOutShape = kShapeResult.value_or(Shape{});
  static constexpr Layout kOutLayout = make_layout<RowMajor>(kOutShape);

  static constexpr auto kGeometryResult =
      impl::make_geometry(kPlan, std::span<const Layout>{kLayouts}, kOutLayout);
#define EINSUM_GEOMETRY_FAILED(code) failed_with(kGeometryResult, errc::code)
  EINSUM_ASSERT_NO_ERROR(EINSUM_GEOMETRY_FAILED)
#undef EINSUM_GEOMETRY_FAILED
  static constexpr impl::Geometry kGeometry = kGeometryResult.value_or(impl::Geometry{});

  static constexpr index_t kMatrixRows = kOutShape.size() >= 1 ? kOutShape[0] : 1;
  static constexpr index_t kMatrixCols = kOutShape.size() == 2 ? kOutShape[1] : 1;

public:
  static constexpr Shape output_shape = kOutShape;
  static constexpr std::size_t output_size = static_cast<std::size_t>(kOutLayout.size());
  static constexpr std::size_t scratch_size =
      static_cast<std::size_t>(kGeometry.scratch_elems);

  using extents_type = ct::extents_of_t<kOutShape>;
  // The same Owned as the runtime path, over a std::array rather than a vector:
  // both sizes are in the type here, so the object touches no allocator at all.
  using storage_type = OwnedArray<value_type, std::max<std::size_t>(output_size, 1)>;

  constexpr explicit CtEinsum(const Ops &...ops) noexcept
      : store_{kOutLayout}, views_{as_const_view(ops)...} {}

  // One plain contraction over packed operands is the whole of what the
  // subscript asked for, and every extent is known: Eigen can unroll it rather
  // than block it, which is what the fixed-size Map is for.
  void eval() noexcept {
    if constexpr (kFixedGemm) {
      constexpr impl::StepGeom step = kGeometry.steps[0];
      impl::FixedMap<value_type, step.m, step.n>{store_.view().data}.noalias() =
          impl::CFixedMap<value_type, step.m, step.k>{views_[0].data} *
          impl::CFixedMap<value_type, step.k, step.n>{views_[1].data};
    } else {
      impl::execute<value_type>(kPlan, kGeometry,
                                std::span<const TensorView<const value_type>>{views_},
                                store_.view(), std::span<value_type>{scratch_});
    }
  }

  [[nodiscard]] constexpr const auto &get_result() const noexcept { return store_.storage(); }

  [[nodiscard]] std::mdspan<const value_type, extents_type> get_result_span() const noexcept {
    return std::mdspan<const value_type, extents_type>{store_.storage().data()};
  }

  // Rank 2 or less, as a Map over the result array: no copy, and every extent
  // is in the type.
  [[nodiscard]] auto result_matrix() const noexcept
    requires(kOutShape.size() <= 2)
  {
    return impl::CFixedMap<value_type, kMatrixRows, kMatrixCols>{store_.storage().data()};
  }

  // The result in the kind the operands were: an Eigen matrix for Eigen
  // operands, an mdspan over this object's own array for mdspan operands.  The
  // mdspan is a view and so costs nothing; the matrix owns its elements,
  // because an Eigen expression pointing into a temporary einsum would not.
  [[nodiscard]] auto result() const noexcept {
    if constexpr (kind == OperandKind::eigen) {
      static_assert(kOutShape.size() <= 2,
                    "einsum<\"...\">: this subscript has a result of rank 3 or more, "
                    "which is no Eigen matrix -- read it with get_result_span(), or "
                    "name your own output with into(...) on the runtime path");
      return matrix_type{result_matrix()};
    } else {
      return get_result_span();
    }
  }

private:
  // Eigen refuses a row-major column and a column-major row, so a degenerate
  // extent picks the order rather than the operands do.
  static constexpr int kEigenOptions =
      (kMatrixRows == 1 && kMatrixCols != 1)  ? Eigen::RowMajor
      : (kMatrixCols == 1)                    ? Eigen::ColMajor
      : std::same_as<ct::first_layout_policy_t<Ops...>, RowMajor> ? Eigen::RowMajor
                                                                  : Eigen::ColMajor;

public:
  using matrix_type = Eigen::Matrix<value_type, static_cast<int>(kMatrixRows),
                                    static_cast<int>(kMatrixCols), kEigenOptions>;

private:
  // as_view() answers the operand's own constness; an input is read-only here
  // whatever the caller handed over.  Not aggregate initialisation, because
  // that would try to build the pointer out of the whole view.
  template <typename Op>
  [[nodiscard]] static constexpr TensorView<const value_type>
  as_const_view(const Op &op) noexcept {
    const auto view = as_view(op);
    return {view.data, view.layout};
  }

  // The fast path applies only where a fixed-size Map is exactly right: one
  // step, nothing summed beforehand, no batch, and all three rectangles packed
  // the way a Matrix<M, N> is.
  static constexpr bool kFixedGemm = [] {
    if (kOperands != 2 || kGeometry.steps.size() != 1) {
      return false;
    }
    if (kGeometry.preps[0].reduced || kGeometry.preps[1].reduced) {
      return false;
    }
    const impl::StepGeom &step = kGeometry.steps[0];
    const auto packed = [](const impl::Slab &slab, const index_t cols) {
      return slab.mappable && !slab.transposed && slab.outer_stride == cols;
    };
    return step.batches == 1 && !step.hadamard && packed(step.l, step.k) &&
           packed(step.r, step.n) && packed(step.out, step.n);
  }();

  storage_type store_;
  std::array<value_type, std::max<std::size_t>(scratch_size, 1)> scratch_{};
  std::array<TensorView<const value_type>, kOperands> views_;
};

// einsum<"ij,jk->ik">(a, b).  The object holds views of the operands, so it
// must not outlive them -- which is what makes it worth naming rather than
// storing.
// No requires-clause on the homogeneity: CtEinsum asserts it, and a
// static_assert says which rule was broken where a failed constraint would only
// say that no overload matched.
template <impl::FixedString S, ct::CStaticOperand... Ops>
[[nodiscard]] constexpr auto einsum(Ops &&...ops) noexcept {
  return CtEinsum<S, std::remove_cvref_t<Ops>...>{ops...};
}

} // namespace einsum
