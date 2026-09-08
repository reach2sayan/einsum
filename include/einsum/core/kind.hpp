#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/util/concepts.hpp"

#include <Eigen/Core>

#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

// What an operand may be, decided by how it is indexed rather than by which
// library it came from: one subscript taking every index, or a chain taking
// one each.
namespace einsum {

namespace impl {

// Which of the pack the "all the same" question is asked against.
template <typename First, typename...> struct first_of {
  using type = First;
};
template <typename... Ops> using first_of_t = typename first_of<Ops...>::type;

// Eigen's own idea of a dense object: the memory has to be there, so a Product
// -- which has no data() -- is not one.
template <typename D>
concept CEigenDense =
    std::derived_from<D, Eigen::DenseBase<D>> && requires(const D &d) {
      typename D::Scalar;
      { d.outerStride() } -> std::convertible_to<Eigen::Index>;
      { d.data() } -> std::convertible_to<const typename D::Scalar *>;
    };

// Eigen's Tensor module by shape, not by name: this header does not include
// <unsupported/Eigen/CXX11/Tensor>, and a caller who wants one already has.
template <typename X>
concept CEigenTensor = requires(const X &x) {
  typename X::Scalar;
  { X::NumDimensions } -> std::convertible_to<int>;
  { X::Layout } -> std::convertible_to<int>;
  { x.dimension(0) } -> std::convertible_to<index_t>;
  { x.data() } -> std::convertible_to<const typename X::Scalar *>;
};

// An mdspan that owns its elements: what a view-family call hands back, and
// contiguous, so the kernels write into it directly.
template <typename X>
concept CMdarray = requires(const X &x) {
  { x.to_mdspan() };
  { x.container() };
  { x.data() };
};

// Named by its operations, so a conforming reimplementation is one too.  The
// rank is asked for rather than searched: a rank-0 mdspan takes no indices.
template <typename X>
concept CMdspanLike = requires(const X &x, std::size_t r) {
  { X::rank() } -> std::convertible_to<std::size_t>;
  { x.extent(r) } -> std::convertible_to<index_t>;
};

} // namespace impl

// x[i, j, ...]: one subscript, every index.  Eigen's (i, j) is the same thing
// in older syntax.
template <typename X>
concept CMultiIndexable = impl::CMdspanLike<std::remove_cvref_t<X>> ||
                          impl::CEigenDense<std::remove_cvref_t<X>> ||
                          impl::CEigenTensor<std::remove_cvref_t<X>>;

namespace impl {

// x[i][j]...: ranges down to a scalar leaf, the depth being the rank.
template <typename X> [[nodiscard]] consteval std::size_t nest_rank() noexcept;

template <typename X> struct nest_leaf {
  using type = X;
};

template <typename X>
  requires std::ranges::range<X>
struct nest_leaf<X> {
  using type = typename nest_leaf<
      std::remove_cvref_t<std::ranges::range_value_t<X>>>::type;
};

template <typename X>
using nest_leaf_t = typename nest_leaf<std::remove_cvref_t<X>>::type;

template <typename X> [[nodiscard]] consteval std::size_t nest_rank() noexcept {
  if constexpr (std::ranges::range<X>) {
    return 1 + nest_rank<std::remove_cvref_t<std::ranges::range_value_t<X>>>();
  } else {
    return 0;
  }
}

} // namespace impl

// Ruled out for anything answering a multidimensional subscript, so an mdspan
// is never mistaken for a one-deep nest.
template <typename X>
concept CNestedIndexable =
    !CMultiIndexable<X> && std::ranges::range<std::remove_cvref_t<X>> &&
    CScalar<impl::nest_leaf_t<X>> &&
    (impl::nest_rank<std::remove_cvref_t<X>>() > 0);

template <typename X>
concept COperand = CMultiIndexable<X> || CNestedIndexable<X>;

// --- what an operand is made of ----------------------------------------------
namespace impl {

// One function, not three specialisations: Eigen satisfies both
// CMultiIndexable and CEigenDense, so specialisations are ambiguous.
template <typename B> [[nodiscard]] consteval auto scalar_probe() noexcept {
  if constexpr (CEigenDense<B> || CEigenTensor<B>) {
    return std::type_identity<typename B::Scalar>{};
  } else if constexpr (CNestedIndexable<B>) {
    return std::type_identity<nest_leaf_t<B>>{};
  } else {
    return std::type_identity<typename B::value_type>{};
  }
}

} // namespace impl

template <COperand X>
using scalar_of_t = std::remove_cv_t<
    typename decltype(impl::scalar_probe<std::remove_cvref_t<X>>())::type>;

// An Eigen vector is rank 1 however it is stored; a nest's rank is its depth.
template <COperand X> [[nodiscard]] consteval std::size_t rank_of() noexcept {
  using B = std::remove_cvref_t<X>;
  if constexpr (impl::CEigenTensor<B>) {
    return static_cast<std::size_t>(B::NumDimensions);
  } else if constexpr (impl::CEigenDense<B>) {
    return B::IsVectorAtCompileTime ? 1 : 2;
  } else if constexpr (CNestedIndexable<B>) {
    return impl::nest_rank<B>();
  } else {
    return static_cast<std::size_t>(B::rank());
  }
}

template <COperand X> inline constexpr std::size_t rank_v = rank_of<X>();

// --- families ----------------------------------------------------------------
// Four answers to "what shall the result be", and a call is in exactly one: an
// Eigen matrix, a Tensor (where a rank-3 result has to live), a nest, or -- for
// the views, which own nothing -- an mdarray.
template <typename X>
concept CEigenFamily = impl::CEigenDense<std::remove_cvref_t<X>>;

// Its own family: a Tensor and a Matrix cannot be one result type.
template <typename X>
concept CTensorFamily = impl::CEigenTensor<std::remove_cvref_t<X>>;

template <typename X>
concept CViewFamily = impl::CMdspanLike<std::remove_cvref_t<X>>;

template <typename X>
concept CNestFamily = CNestedIndexable<X>;

// Can the contraction be written straight into this result's storage?  Anything
// else is filled through scratch and scattered.  Both entry points ask it.
template <typename R>
concept CDirectWritable = impl::CEigenDense<R> || impl::CEigenTensor<R> ||
                          impl::CMdarray<R> || rank_v<R> == 1;

// One family and one scalar across the call; ranks may differ, so not "the
// same type".
template <typename... Ops>
concept CSameFamily =
    sizeof...(Ops) > 0 && (COperand<Ops> && ...) &&
    ((CEigenFamily<Ops> && ...) || (CTensorFamily<Ops> && ...) ||
     (CViewFamily<Ops> && ...) || (CNestFamily<Ops> && ...)) &&
    (std::same_as<scalar_of_t<impl::first_of_t<Ops...>>, scalar_of_t<Ops>> &&
     ...);

// The rank the result is built at when the subscript is not in a type: the
// only one every family can represent.
template <typename... Ops>
inline constexpr std::size_t widest_rank_v = std::max({rank_v<Ops>...});

namespace impl {
// And the operand that has it: what a family carrying its rank in its type
// builds its result from.
template <typename... Ops> struct widest_of;
template <typename A> struct widest_of<A> {
  using type = A;
};
template <typename A, typename B, typename... Rest>
struct widest_of<A, B, Rest...> {
  using type = typename widest_of<
      std::conditional_t<(rank_v<B> > rank_v<A>), std::remove_cvref_t<B>,
                         std::remove_cvref_t<A>>,
      Rest...>::type;
};
} // namespace impl

template <typename... Ops>
using widest_of_t = typename impl::widest_of<std::remove_cvref_t<Ops>...>::type;

} // namespace einsum
