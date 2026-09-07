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
template <impl::FixedString S> class StaticEinsum : public Einsum {
  static constexpr Subscripts kSubscripts = ct::subscripts<S>::value;

  static constexpr auto kPlanResult = impl::make_plan(kSubscripts);
#define EINSUM_PLAN_FAILED(code) failed_with(kPlanResult, errc::code)
  EINSUM_ASSERT_NO_ERROR(EINSUM_PLAN_FAILED)
#undef EINSUM_PLAN_FAILED

public:
  StaticEinsum() noexcept
      : Einsum{einsum::impl::einsum_access::make<einsum::impl::HeapScratch>(
            kPlanResult.value_or(Plan{}))} {}

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

  // einsum<"ij,jk->ik">(a, b)      -> result<R>, by value
  // einsum<"ij,jk->ik">(a, b, out) -> result<void>, moved into out
  template <typename... A> [[nodiscard]] auto operator()(A &&...args) const {
    constexpr std::size_t n = sizeof...(A);
    if constexpr (n == operand_count() + 1) {
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
        return Einsum::operator()(args...);
      }
    }
  }

private:
  // --- the constant lowering -------------------------------------------------
  // Every extent is in a type, so the shape, the output layout, the Geometry
  // and the scratch size are all constants, and the scratch is an array in this
  // frame rather than anything the object had to allocate.
  template <typename... Ops> struct Lowered {
    static constexpr std::array<Layout, sizeof...(Ops)> layouts{
        ct::layout_of<Ops>()...};
    static constexpr Shape shape =
        impl::infer_output_shape(kPlanResult.value_or(Plan{}),
                                 std::span<const Layout>{layouts})
            .value_or(Shape{});

    // The view family's result is an mdarray either way; here its extents are
    // known, so it is one over a std::array and the call allocates nothing.
    using R = std::conditional_t<
        CViewFamily<einsum::impl::first_of_t<Ops...>>,
        ct::static_mdarray_t<scalar_of_t<einsum::impl::first_of_t<Ops...>>, shape>,
        result_of_t<Ops...>>;
    static constexpr bool direct = impl::CEigenDense<R> || impl::CEigenTensor<R> ||
                                   impl::CMdarray<R> || rank_v<R> == 1;
    static constexpr Layout out_layout =
        einsum::impl::layout_of_result<R>(shape);
    static constexpr impl::Geometry geometry =
        impl::make_geometry(kPlanResult.value_or(Plan{}),
                            std::span<const Layout>{layouts}, out_layout)
            .value_or(impl::Geometry{});

    // The scratch the kernels want, then a packed copy of any operand they
    // cannot address, then the output when it is one of those.
    static constexpr std::array<bool, sizeof...(Ops)> gathered{
        !CContiguous<Ops>...};
    static constexpr index_t packed_bytes = [] {
      index_t at = geometry.scratch_elems;
      for (const auto i : std::views::iota(std::size_t{0}, sizeof...(Ops))) {
        if (gathered[i]) {
          at += layouts[i].size();
        }
      }
      return at;
    }();
    static constexpr index_t total =
        direct ? packed_bytes : packed_bytes + einsum::impl::product(shape);

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

  template <typename... Ops>
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
      boost::container::static_vector<TensorView<const T>, kMaxOperands> views;
      std::size_t next = 0;
      index_t cursor = L::geometry.scratch_elems;
      const auto prepare = [&](const auto &op) {
        const T *base = nullptr;
        if constexpr (CContiguous<decltype(op)>) {
          base = einsum::impl::data_of(op);
        } else {
          T *const packed = scratch.data() + cursor;
          gather(op, packed, L::layouts[next].shape);
          cursor += L::layouts[next].size();
          base = packed;
        }
        views.push_back(TensorView<const T>{base, L::layouts[next]});
        ++next;
      };
      (prepare(ops), ...);

      T *target_data = nullptr;
      if constexpr (L::direct) {
        target_data = out->data();
      } else {
        target_data = scratch.data() + cursor;
      }
      einsum::impl::execute<T>(
          kPlanResult.value_or(Plan{}), L::geometry,
          std::span<const TensorView<const T>>{views},
          TensorView<T>{target_data, L::out_layout},
          std::span<T>{scratch}.first(
              static_cast<std::size_t>(L::geometry.scratch_elems)));
      if constexpr (!L::direct) {
        const auto fitted = einsum::impl::fit_shape(L::shape, rank_v<R>);
        scatter(static_cast<const T *>(target_data), *out, *fitted);
      }
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
template <impl::FixedString S> [[nodiscard]] StaticEinsum<S> einsum() noexcept {
  return {};
}

} // namespace einsum
