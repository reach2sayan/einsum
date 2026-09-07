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
// library it came from.  Two accessor models, because there are two ways C++
// spells "the element at (i, j)": one subscript taking every index, and a chain
// of subscripts taking one each.  Everything else about an operand -- its
// extents, its scalar, whether its memory is contiguous -- is read through
// whichever of the two it answers to.
namespace einsum {

namespace impl {

// Which of the pack the "all the same" question is asked against.  The only
// type computation here, and only because a type is what it answers.
template <typename First, typename...> struct first_of {
  using type = First;
};
template <typename... Ops> using first_of_t = typename first_of<Ops...>::type;

// Eigen's own idea of a dense object: the memory has to be there.  A Product
// has no data() and is not one.
template <typename D>
concept CEigenDense =
    std::derived_from<D, Eigen::DenseBase<D>> && requires(const D &d) {
      typename D::Scalar;
      { d.outerStride() } -> std::convertible_to<Eigen::Index>;
      { d.data() } -> std::convertible_to<const typename D::Scalar *>;
    };

// Eigen's unsupported Tensor module, recognised by its shape rather than by
// name: this header does not include <unsupported/Eigen/CXX11/Tensor>, because
// most callers do not want it, and a caller who does has already included it.
template <typename X>
concept CEigenTensor = requires(const X &x) {
  typename X::Scalar;
  { X::NumDimensions } -> std::convertible_to<int>;
  { X::Layout } -> std::convertible_to<int>;
  { x.dimension(0) } -> std::convertible_to<index_t>;
  { x.data() } -> std::convertible_to<const typename X::Scalar *>;
};

// An mdarray: an mdspan that owns its elements.  It is what a view-family call
// hands back, and it is contiguous, so the kernels write into it directly.
template <typename X>
concept CMdarray = requires(const X &x) {
  { x.to_mdspan() };
  { x.container() };
  { x.data() };
};

// An mdspan, or anything that describes itself the way one does.  Naming the
// operations rather than the class template lets a conforming reimplementation
// through.
// The rank is asked for rather than searched: a rank-0 mdspan or mdarray takes
// no indices at all, so probing x[i] for i = 1.. can never find it.
template <typename X>
concept CMdspanLike = requires(const X &x, std::size_t r) {
  { X::rank() } -> std::convertible_to<std::size_t>;
  { x.extent(r) } -> std::convertible_to<index_t>;
};

} // namespace impl

// --- the accessor models -----------------------------------------------------
// x[i, j, ...]: one subscript, every index.  Eigen's dense objects and its
// Tensors belong here too -- their accessor is spelled (i, j), which is the
// same thing wearing older syntax.
template <typename X>
concept CMultiIndexable = impl::CMdspanLike<std::remove_cvref_t<X>> ||
                          impl::CEigenDense<std::remove_cvref_t<X>> ||
                          impl::CEigenTensor<std::remove_cvref_t<X>>;

namespace impl {

// x[i][j]...: a range whose elements are ranges, down to a scalar leaf.  The
// depth is the rank, and it is found by descending.
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

// A nest is a range whose leaf is a scalar.  Ruled out for anything that
// already answers a multidimensional subscript, so an mdspan is never mistaken
// for a one-deep nest.
template <typename X>
concept CNestedIndexable =
    !CMultiIndexable<X> && std::ranges::range<std::remove_cvref_t<X>> &&
    CScalar<impl::nest_leaf_t<X>> &&
    (impl::nest_rank<std::remove_cvref_t<X>>() > 0);

template <typename X>
concept COperand = CMultiIndexable<X> || CNestedIndexable<X>;

// --- what an operand is made of ----------------------------------------------
namespace impl {

// One function rather than three partial specialisations: Eigen satisfies both
// CMultiIndexable and CEigenDense, so specialising on each of them is ambiguous
// where an if-constexpr chain simply has an order.
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

// An Eigen vector is rank 1 however it is stored; everything else says its own
// rank, and a nest's is how deep it goes.
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
// Four, because there are four answers to "what shall the result be": an Eigen
// matrix, an Eigen Tensor (which is where a rank-3 result has to live, since a
// matrix stops at two), a nest of ranges, or -- for a view, which can own
// nothing itself -- the owning member of its own family, an mdarray.  A call is
// in exactly one.
template <typename X>
concept CEigenFamily = impl::CEigenDense<std::remove_cvref_t<X>>;

// Its own family, not Eigen's: a Tensor and a Matrix cannot be the same result
// type, and a rank-3 result has to be a Tensor.
template <typename X>
concept CTensorFamily = impl::CEigenTensor<std::remove_cvref_t<X>>;

template <typename X>
concept CViewFamily = impl::CMdspanLike<std::remove_cvref_t<X>>;

template <typename X>
concept CNestFamily = CNestedIndexable<X>;

// Can the contraction be written straight into this result's own storage?
// True when its elements are one contiguous run the executor can address: the
// three owning families that guarantee it, plus any rank-1 result, whose single
// axis is contiguous whatever holds it.  Anything else is filled through a
// scratch buffer and scattered afterwards.
//
// Named once because both entry points ask it -- BasicEinsum::evaluate and
// StaticEinsum::Lowered -- and a family added to one spelling but not the other
// would silently scatter into a buffer the caller never reads.
template <typename R>
concept CDirectWritable = impl::CEigenDense<R> || impl::CEigenTensor<R> ||
                          impl::CMdarray<R> || rank_v<R> == 1;

// One family and one scalar across the call.  Ranks may differ -- "ij,j->i" is
// a matrix and a vector -- so this is deliberately not "the same type".
template <typename... Ops>
concept CSameFamily =
    sizeof...(Ops) > 0 && (COperand<Ops> && ...) &&
    ((CEigenFamily<Ops> && ...) || (CTensorFamily<Ops> && ...) ||
     (CViewFamily<Ops> && ...) || (CNestFamily<Ops> && ...)) &&
    (std::same_as<scalar_of_t<impl::first_of_t<Ops...>>, scalar_of_t<Ops>> &&
     ...);

// The rank the result is built at when the subscript is not in a type: the
// widest operand, which is the only rank every family can always represent.
template <typename... Ops>
inline constexpr std::size_t widest_rank_v = std::max({rank_v<Ops>...});

namespace impl {
// And the operand that has it -- which is the type a family whose rank lives in
// its type (a std::array nest, an Eigen Tensor) has to build its result from,
// since neither can be spelled here without naming the library that owns it.
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
