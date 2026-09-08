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

// The compile-time half of the entry point.  The same object and the same
// evaluation as the runtime one; the only difference is that the subscript is a
// template argument, so every way it can be wrong is a static_assert carrying
// the sentence the runtime error would have carried -- and so the operand count
// is a constant, which is what lets this one also take an output.
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

  // A constant here, unlike on the runtime path, which is the whole reason the
  // output form below can exist: one more argument than the subscript names is
  // an output, and nothing has to guess.
  [[nodiscard]] static constexpr std::size_t operand_count() noexcept {
    return kSubscripts.operands.size();
  }

  // Is this call's whole lowering a constant?  Only then is there anything the
  // compile-time path can do that the runtime one cannot.
  template <typename... Ops>
  [[nodiscard]] static consteval bool lowers_statically() noexcept {
    return ct::all_static<Ops...>() && sizeof...(Ops) == operand_count();
  }

  // einsum<"ij,jk->ik">(a, b)           -> result<R>, by value
  // einsum<"ij,jk->ik">(a, b, out)      -> result<void>, moved into out
  // einsum<"ij,jk->ik">(a, b, out(x))   -> the same, spelled as the runtime
  //                                        path has to spell it
  //
  // The count is a constant here, so the untagged form is unambiguous and stays
  // -- but out() is accepted too, so a call moves between the two paths without
  // being rewritten.
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
        // by_value, not operator(): the arity is a constant here and has
        // already settled that this is not the output form, and the runtime
        // operator() would ask again -- by constness, which a mutable operand
        // would answer wrongly.
        return this->by_value(args...);
      }
    }
  }

private:
  // --- the constant lowering -------------------------------------------------
  // Every extent is in a type, so the shape, the output layout, the Geometry
  // and the scratch size are all constants, and the scratch is an array in this
  // frame rather than anything the object had to allocate.
  template <COperand... Ops>
    requires(ct::all_static<Ops...>())
  struct Lowered {
    static constexpr std::array<impl::Layout, sizeof...(Ops)> layouts{
        ct::layout_of<Ops>()...};

    // A '...' stands for as many axes as the operands have, and here that is a
    // constant, so the expanded subscript and the plan built from it are too.
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

    // The view family's result is an mdarray either way; here its extents are
    // known, so it is one over a std::array and the call allocates nothing.
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

    // The scratch the kernels want, then a packed copy of any operand they
    // cannot address, then the output when it is one of those -- laid out by
    // the same function the runtime path uses, but here at compile time, so
    // `total` sizes an array in the frame and the call allocates nothing.
    static constexpr std::array<bool, sizeof...(Ops)> gathered{
        !impl::CContiguous<Ops>...};
    static constexpr einsum::impl::ScratchMap map = einsum::impl::scratch_offsets(
        std::span<const impl::Layout>{layouts}, std::span<const bool>{gathered},
        geometry.scratch_elems, shape, direct);
    static constexpr index_t total = map.total;

    // One plain mappable GEMM, nothing packed and nothing summed first: the
    // case Eigen can unroll rather than block, which is the whole point of
    // knowing the extents this early.
    static constexpr bool fixed_gemm = [] {
      if (sizeof...(Ops) != 2 || geometry.steps.size() != 1 || !direct) {
        return false;
      }
      if (geometry.preps[0].reduced || geometry.preps[1].reduced) {
        return false;
      }
      if (std::ranges::any_of(gathered, [](const bool g) { return g; })) {
        return false;
      }
      const impl::StepGeom &step = geometry.steps[0];
      // Packed in whichever order it is stored in: a row-major rectangle's
      // outer stride is its column count, a column-major one's is its row
      // count.  Either is a plain fixed-size map -- requiring row-major would
      // miss Eigen's own default order.
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
      // blocked.  Straight into the result's own storage.
      constexpr einsum::impl::StepGeom step = L::geometry.steps[0];
      // Each side keeps the order it is stored in; `transposed` is make_slab's
      // word for "column-major here".
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
      // than a pooled block and with offsets that are constants.
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
  // Whether this call is the one Eigen can unroll: every extent in a type, one
  // mappable GEMM step, nothing packed, nothing summed first, and a result the
  // product can be written straight into.  Public so a test can assert it
  // rather than infer it from a timing.
  template <typename... Ops>
  [[nodiscard]] static consteval bool unrolls() noexcept {
    if constexpr (CSameFamily<Ops...> && lowers_statically<Ops...>()) {
      return Lowered<Ops...>::fixed_gemm;
    } else {
      return false;
    }
  }
};

// einsum<"ij,jk->ik">() -- a function template rather than a variable template,
// so that it and the runtime einsum(std::string_view) are overloads of one name
// rather than two different kinds of entity, which C++ will not have.
template <impl::FixedString S, path P = path::greedy>
[[nodiscard]] StaticEinsum<S, P> einsum() noexcept {
  return {};
}

// --- the subscript as an argument --------------------------------------------
// A function argument is never a constant expression, so einsum("ij,jk->ik")
// cannot be the compile-time call however it is written -- the subscript has to
// reach a template parameter, and only a literal operator template can put it
// there while still looking like a string.  That is the whole of what _ct is:
// an empty type carrying S, so that the two paths are one name taking one
// string and differ by a suffix rather than by a syntax.
namespace ct {

template <einsum::impl::FixedString S, path P = path::greedy>
struct Subscript {
  // "ab,bc,cd->ad"_ct.with<path::sequential>() -- the contraction order chosen
  // where the subscript is written, since it cannot be a second argument for
  // the same reason the subscript cannot be a first one.
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

// einsum("ij,jk->ik"_ct) -- the same object einsum<"ij,jk->ik">() answers, and
// what the suffix exists for.
template <impl::FixedString S, path P>
[[nodiscard]] StaticEinsum<S, P> einsum(ct::Subscript<S, P>) noexcept {
  return {};
}

} // namespace einsum
