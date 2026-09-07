#pragma once

#include "einsum/core/kind.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"

#include <Eigen/Core>
#include <experimental/mdarray>
#include <experimental/mdspan>

#include <array>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsum {

// --- what a call hands back
// --------------------------------------------------- The result is the
// operands' own type, so there is no result kind to choose and no accessor to
// remember: an einsum over Eigen matrices answers an Eigen matrix, one over a
// vector of vectors answers a vector of vectors.
//
// The exception is the view family -- mdspan, and anything else that only
// borrows its memory -- which cannot own a result at all.  Those get a nest of
// std::vector of the same scalar and rank, returned by value like every other
// result: the library never owns an output, so there is nothing whose lifetime
// a caller has to reason about.
namespace impl {

// The output shape, fitted to the rank the result type has.  A lower rank is
// padded with leading extents of 1 -- which changes nothing about the elements,
// only how they are described -- and a higher one cannot be represented at all.
[[nodiscard]] constexpr result<Shape>
fit_shape(const Shape &shape, const std::size_t rank) noexcept {
  if (shape.size() > rank) {
    return fail(errc::rank_mismatch);
  }
  Shape fitted;
  for (std::size_t i = shape.size(); i < rank; ++i) {
    fitted.push_back(1);
  }
  for (const index_t extent : shape) {
    fitted.push_back(extent);
  }
  return fitted;
}

// The one place a result is created.  Eigen sizes itself from the shape; a
// nest resizes level by level; a std::array nest has its extents in its type
// and can only check them.
template <typename X>
[[nodiscard]] result<X> make_like(const Shape &shape) noexcept;

template <std::size_t D, std::size_t R, typename X>
[[nodiscard]] constexpr result<void> shape_nest(X &x,
                                                const Shape &shape) noexcept {
  if constexpr (D == R) {
    return {};
  } else {
    const auto want = static_cast<std::size_t>(shape[D]);
    if constexpr (requires { x.resize(want); }) {
      x.resize(want);
    } else if (std::ranges::size(x) != want) {
      // A std::array nest carries its extents in its type; the subscript has
      // to agree with them rather than the other way round.
      return fail(errc::output_mismatch);
    }
    for (auto &row : x) {
      if (const auto ok = shape_nest<D + 1, R>(row, shape); !ok) {
        return ok;
      }
    }
    return {};
  }
}

template <typename X>
[[nodiscard]] result<X> make_like(const Shape &shape) noexcept {
  using B = std::remove_cvref_t<X>;
  if constexpr (CMdarray<B>) {
    const auto fitted = fit_shape(shape, rank_v<B>);
    if (!fitted) {
      return propagate<X>(fitted.error());
    }
    if constexpr (rank_v<B> == 0) {
      // A rank-0 mdarray holds one element and has no extents to be given.
      return B{typename B::extents_type{}};
    } else {
      std::array<std::size_t, rank_v<B>> ext{};
      std::ranges::transform(*fitted, ext.begin(),
                             [](const index_t e) { return static_cast<std::size_t>(e); });
      return B{typename B::extents_type{ext}};
    }
  } else if constexpr (CEigenTensor<B>) {
    // A Tensor carries its rank in its type, so a shorter output shape is
    // padded and a longer one has nowhere to go.
    const auto fitted = fit_shape(shape, rank_v<B>);
    if (!fitted) {
      return propagate<X>(fitted.error());
    }
    typename B::Dimensions dims;
    for (const auto i : std::views::iota(std::size_t{0}, rank_v<B>)) {
      dims[i] = static_cast<typename B::Index>((*fitted)[i]);
    }
    B out(dims); // parentheses: braces would reach the variadic-extent ctor
    out.setZero();
    return out;
  } else if constexpr (CEigenDense<B>) {
    if (shape.size() > 2) {
      return fail(errc::rank_mismatch);
    }
    const auto rows =
        static_cast<Eigen::Index>(shape.size() >= 1 ? shape[0] : 1);
    const auto cols =
        static_cast<Eigen::Index>(shape.size() == 2 ? shape[1] : 1);
    // A fixed-size Eigen type cannot be resized into agreement; it can only be
    // the right size already.
    if constexpr (B::RowsAtCompileTime != Eigen::Dynamic) {
      if (B::RowsAtCompileTime != rows) {
        return fail(errc::output_mismatch);
      }
    }
    if constexpr (B::ColsAtCompileTime != Eigen::Dynamic) {
      if (B::ColsAtCompileTime != cols) {
        return fail(errc::output_mismatch);
      }
    }
    B out;
    if constexpr (B::SizeAtCompileTime == Eigen::Dynamic) {
      if constexpr (B::IsVectorAtCompileTime) {
        out.resize(rows * cols);
      } else {
        out.resize(rows, cols);
      }
    }
    out.setZero();
    return out;
  } else {
    const auto fitted = fit_shape(shape, rank_v<B>);
    if (!fitted) {
      return propagate<X>(fitted.error());
    }
    B out{};
    return shape_nest<0, rank_v<B>>(out, *fitted).transform([&] noexcept {
      return std::move(out);
    });
  }
}

} // namespace impl

namespace impl {

// A nest of std::vector R deep over T; R == 0 is the scalar itself.
template <typename T, std::size_t R> struct nest_of {
  using type = std::vector<typename nest_of<T, R - 1>::type>;
};
template <typename T> struct nest_of<T, 0> {
  using type = T;
};
template <typename T, std::size_t R>
using nest_of_t = typename nest_of<T, R>::type;

} // namespace impl

// The result type of a call over these operands.  It is fixed by the operand
// types alone, because on the runtime path the subscript is not a type and the
// return type still has to be one -- so the rank is the widest operand's and
// the true output shape is fitted to it: a lower rank is padded with leading
// extents of 1, and a higher one is a rank_mismatch the `out&` form can name
// its way out of.
//
// Eigen answers an Eigen matrix in the first operand's storage order (rank 0,
// 1 and 2 being 1x1, Nx1 and MxN by Eigen's own convention); the other two
// families answer a nest of vectors, because a view cannot own a result and a
// nest already is one.
namespace impl {

template <typename X>
inline constexpr int eigen_order_of =
    std::remove_cvref_t<X>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;

template <typename... Ops>
[[nodiscard]] consteval auto result_probe() noexcept {
  using First = std::remove_cvref_t<first_of_t<Ops...>>;
  using T = scalar_of_t<First>;
  constexpr std::size_t kRank = widest_rank_v<Ops...>;
  if constexpr (CEigenFamily<First>) {
    return std::type_identity<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                            eigen_order_of<First>>>{};
  } else if constexpr (CTensorFamily<First>) {
    // A Tensor's rank is in its type, and that type cannot be rebuilt here
    // without naming Eigen's Tensor header, so the result is the widest operand
    // itself.  An output deeper than that is rank_mismatch, as for array nests.
    return std::type_identity<widest_of_t<Ops...>>{};
  } else if constexpr (CNestFamily<First> &&
                       rank_v<widest_of_t<Ops...>> == kRank) {
    // A nest that is already the right depth answers its own type, which is
    // what makes an einsum over std::array nests give back a std::array nest.
    return std::type_identity<widest_of_t<Ops...>>{};
  } else {
    // A view cannot own a result, and a nest shallower than the output needs a
    // deeper one than it is; both answer the vector nest of that rank.
    return std::type_identity<nest_of_t<T, kRank>>{};
  }
}

} // namespace impl

template <typename... Ops>
  requires CSameFamily<Ops...>
using result_of_t = typename decltype(impl::result_probe<Ops...>())::type;

} // namespace einsum
