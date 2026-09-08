#pragma once

#include "einsum/core/einsum_object.hpp"
#include "einsum/core/kernels.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/ct/static_shape.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/fixed_string.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <tuple>
#include <utility>

namespace einsum {

// The same object and evaluation as the runtime path, but the subscript is a
// template argument: every way it can be wrong is a static_assert carrying the
// sentence the runtime error would have, and the operand count is a constant,
// which is what lets this one take an untagged output.
template <impl::FixedString S, path P = path::greedy>
class StaticEinsum : public Einsum {
  friend struct einsum::impl::einsum_access;

  static constexpr Subscripts kSubscripts = ct::subscripts<S>::value;

  static constexpr auto kPlanResult = impl::make_plan(kSubscripts);
#define EINSUM_PLAN_FAILED(code) failed_with(kPlanResult, errc::code)
  EINSUM_ASSERT_NO_ERROR(EINSUM_PLAN_FAILED)
#undef EINSUM_PLAN_FAILED

public:
  StaticEinsum() noexcept
      : Einsum{einsum::impl::einsum_access::make<einsum::impl::HeapScratch>(
            kPlanResult.value_or(Plan{}), P)} {}

  // A constant here, unlike on the runtime path: one more argument than the
  // subscript names is an output, and nothing has to guess.
  [[nodiscard]] static constexpr std::size_t operand_count() noexcept {
    return kSubscripts.operands.size();
  }

  // Is this call's whole lowering a constant?
  template <typename... Ops>
  [[nodiscard]] static consteval bool lowers_statically() noexcept {
    return ct::all_static<Ops...>() && sizeof...(Ops) == operand_count();
  }

  // (a, b)         -> result<R>
  // (a, b, out)     -> result<void>, moved into out
  // (a, b, out(x))  -> the same, spelled as the runtime path must spell it
  template <typename... A> [[nodiscard]] auto operator()(A &&...args) const {
    constexpr std::size_t n = sizeof...(A);
    if constexpr (impl::is_rt_out_form<A...>()) {
      static_assert(n == operand_count() + 1,
                    "einsum<\"...\">: out(x) is one argument past the operands "
                    "the subscript names, and this call has a different number "
                    "of them");
      return this->untag_output(std::forward_as_tuple(EINSUM_FWD(args)...),
                                std::make_index_sequence<n - 1>{});
    } else if constexpr (n == operand_count() + 1) {
      return this->with_output(std::forward_as_tuple(EINSUM_FWD(args)...),
                               std::make_index_sequence<n - 1>{});
    } else {
      static_assert(
          n == operand_count(),
          "einsum<\"...\">: the subscript names a different number of operands "
          "than the call passes (one more than that is an output)");
      static_assert(
          CSameFamily<A...>,
          "einsum<\"...\">: one call's operands must be one family and one "
          "scalar "
          "-- all Eigen objects, all mdspans or spans, or all nested ranges; "
          "their ranks may differ");
      if constexpr (!CSameFamily<A...>) {
        return result<void>{};
      } else if constexpr (lowers_statically<A...>()) {
        return statically(args...);
      } else {
        // by_value, not operator(): the arity has already settled that this is
        // not the output form, and operator() would ask again.
        return this->by_value(args...);
      }
    }
  }

private:
  // Every extent is in a type, so the shape, the output layout, the Geometry
  // and the scratch size are all constants.
  template <COperand... Ops>
    requires(ct::all_static<Ops...>())
  struct Lowered {
    static constexpr std::array<impl::Layout, sizeof...(Ops)> layouts{
        ct::layout_of<Ops>()...};

    // A '...' stands for as many axes as the operands have -- a constant here,
    // so the expanded subscript and its plan are constants too.
    static constexpr std::array<std::uint8_t, sizeof...(Ops)> ranks{
        static_cast<std::uint8_t>(rank_v<Ops>)...};
    static constexpr auto kExpanded =
        einsum::impl::expand(kSubscripts, std::span<const std::uint8_t>{ranks});
#define EINSUM_EXPAND_FAILED(code) failed_with(kExpanded, errc::code)
    EINSUM_ASSERT_NO_ERROR(EINSUM_EXPAND_FAILED)
#undef EINSUM_EXPAND_FAILED

    static constexpr auto kLowered = einsum::impl::make_plan(kExpanded.value_or(Subscripts{}));
#define EINSUM_LOWERED_FAILED(code) failed_with(kLowered, errc::code)
    EINSUM_ASSERT_NO_ERROR(EINSUM_LOWERED_FAILED)
#undef EINSUM_LOWERED_FAILED
    static constexpr Plan plan = kLowered.value_or(Plan{});

    static constexpr Shape shape =
        impl::infer_output_shape(plan, std::span<const impl::Layout>{layouts})
            .value_or(Shape{});

    // The view family's result is an mdarray either way; here over a
    // std::array, so the call allocates nothing.
    using R = std::conditional_t<
        CViewFamily<einsum::impl::first_of_t<Ops...>>,
        ct::static_mdarray_t<scalar_of_t<einsum::impl::first_of_t<Ops...>>, shape>,
        result_of_t<Ops...>>;
    static constexpr bool direct = CDirectWritable<R>;
    static constexpr impl::Layout out_layout =
        einsum::impl::layout_of_result<R>(shape);
    static constexpr impl::Geometry geometry =
        impl::make_geometry(plan, std::span<const impl::Layout>{layouts}, out_layout, P)
            .value_or(impl::Geometry{});

    // Laid out by the same function the runtime path uses, but at compile
    // time, so `total` sizes an array in the frame.
    static constexpr std::array<bool, sizeof...(Ops)> gathered{
        !impl::CContiguous<Ops>...};
    static constexpr einsum::impl::ScratchMap map = einsum::impl::scratch_offsets(
        std::span<const impl::Layout>{layouts}, std::span<const bool>{gathered},
        geometry.scratch_elems, shape, direct);
    static constexpr index_t total = map.total;

    // One mappable GEMM, nothing packed or summed first: the case Eigen
    // unrolls rather than blocks.
    static constexpr bool fixed_gemm = [] {
      if (sizeof...(Ops) != 2 || geometry.steps.size() != 1 || !direct) {
        return false;
      }
      if (geometry.preps[0].reduced || geometry.preps[1].reduced) {
        return false;
      }
      if (std::ranges::contains(gathered, true)) {
        return false;
      }
      const impl::StepGeom &step = geometry.steps[0];
      // Packed in whichever order it is stored in: a row-major rectangle's
      // outer stride is its column count, a column-major one's its row count.
      // Requiring row-major would miss Eigen's own default order.
      const auto packed = [](const impl::Slab &slab, const index_t rows,
                             const index_t cols) {
        return slab.mappable &&
               slab.outer_stride == (slab.transposed ? rows : cols);
      };
      return step.batches() == 1 && !step.hadamard() &&
             packed(step.l, step.m(), step.k()) &&
             packed(step.r, step.k(), step.n()) &&
             packed(step.out, step.m(), step.n());
    }();
  };

  template <COperand... Ops>
    requires(ct::all_static<Ops...>())
  [[nodiscard]] auto statically(const Ops &...ops) const {
    using L = Lowered<Ops...>;
    using R = typename L::R;
    using T = scalar_of_t<einsum::impl::first_of_t<Ops...>>;

    auto out = einsum::impl::make_like<R>(L::shape);
    if (!out) {
      return propagate<R>(out.error());
    }

    std::array<T, static_cast<std::size_t>(L::total) == 0
                      ? 1
                      : static_cast<std::size_t>(L::total)>
        scratch{};

    if constexpr (L::fixed_gemm) {
      // Every extent in the type, so the product is unrolled rather than
      // blocked, straight into the result's own storage.  Each side keeps the
      // order it is stored in; `transposed` is make_slab's word for
      // "column-major here".
      constexpr einsum::impl::StepGeom step = L::geometry.steps[0];
      constexpr bool l_row = !step.l.transposed;
      constexpr bool r_row = !step.r.transposed;
      constexpr bool o_row = !step.out.transposed;
      // A lambda rather than pack indexing, which is C++26.
      const auto product = [&](const auto &left, const auto &right) noexcept {
        einsum::impl::FixedMap<T, step.m(), step.n(), o_row>{out->data()}
            .noalias() =
            einsum::impl::CFixedMap<T, step.m(), step.k(), l_row>{
                einsum::impl::data_of(left)} *
            einsum::impl::CFixedMap<T, step.k(), step.n(), r_row>{
                einsum::impl::data_of(right)};
      };
      product(ops...);
      return result<R>{std::move(*out)};
    } else {
      // The same contraction the runtime path runs, over a frame array rather
      // than a pooled block.
      static constexpr auto kFitted =
          einsum::impl::fit_shape(L::shape, rank_v<R>).value_or(Shape{});
      einsum::impl::contract_into<T>(L::plan, L::geometry,
                                     std::span<const impl::Layout>{L::layouts},
                                     L::out_layout, scratch.data(), L::map, *out,
                                     kFitted, ops...);
      return result<R>{std::move(*out)};
    }
  }

public:
  // Public so a test asserts the unrolled case rather than timing for it.
  template <typename... Ops>
  [[nodiscard]] static consteval bool unrolls() noexcept {
    if constexpr (CSameFamily<Ops...> && lowers_statically<Ops...>()) {
      return Lowered<Ops...>::fixed_gemm;
    } else {
      return false;
    }
  }
};

template <impl::FixedString S, path P = path::greedy>
[[nodiscard]] StaticEinsum<S, P> einsum() noexcept {
  return {};
}

// A function argument is never a constant expression, so only a literal
// operator template gets the subscript into a template parameter while still
// looking like a string.
namespace ct {

template <einsum::impl::FixedString S, path P = path::greedy>
struct Subscript {
  // The order chosen where the subscript is written, since it cannot be an
  // argument for the same reason the subscript cannot.
  template <path Q>
  [[nodiscard]] consteval Subscript<S, Q> with() const noexcept {
    return {};
  }
};

} // namespace ct

inline namespace literals {

template <impl::FixedString S>
[[nodiscard]] consteval ct::Subscript<S> operator""_ct() noexcept {
  return {};
}

} // namespace literals

// einsum("ij,jk->ik"_ct) -- the same object einsum<"ij,jk->ik">() answers.
template <impl::FixedString S, path P>
[[nodiscard]] StaticEinsum<S, P> einsum(ct::Subscript<S, P>) noexcept {
  return {};
}

} // namespace einsum
