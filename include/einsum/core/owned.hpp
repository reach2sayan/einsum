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
#include <algorithm>
#include <iterator>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsum {

// The result is the operands' own type, except for the view family, which
// cannot own one and gets the owning member of its family instead.
namespace impl {

// Fitted to the result type's rank: padded with leading extents of 1, which
// changes only how the elements are described.  A higher rank cannot be.
[[nodiscard]] constexpr result<Shape>
fit_shape(const Shape &shape, const std::size_t rank) noexcept {
  if (shape.size() > rank) {
    return fail(errc::rank_mismatch);
  }
  Shape fitted;
  std::ranges::fill_n(std::back_inserter(fitted),
                      static_cast<std::ptrdiff_t>(rank - shape.size()),
                      index_t{1});
  std::ranges::copy(shape, std::back_inserter(fitted));
  return fitted;
}

// The one place a result is created.
template <COperand X>
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
      // A std::array nest's extents are in its type; the subscript must agree.
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

template <COperand X>
[[nodiscard]] result<X> make_like(const Shape &shape) noexcept {
  using B = std::remove_cvref_t<X>;
  if constexpr (CMdarray<B>) {
    const auto fitted = fit_shape(shape, rank_v<B>);
    if (!fitted) {
      return propagate<X>(fitted.error());
    }
    if constexpr (rank_v<B> == 0) {
      return B{typename B::extents_type{}};
    } else {
      std::array<std::size_t, rank_v<B>> ext{};
      std::ranges::transform(*fitted, ext.begin(), [](const index_t e) {
        return static_cast<std::size_t>(e);
      });
      return B{typename B::extents_type{ext}};
    }
  } else if constexpr (CEigenTensor<B>) {
    const auto fitted = fit_shape(shape, rank_v<B>);
    if (!fitted) {
      return propagate<X>(fitted.error());
    }
    typename B::Dimensions dims;
    std::ranges::transform(*fitted, dims.begin(), [](const index_t e) {
      return static_cast<typename B::Index>(e);
    });
    B out(dims); // parentheses: braces would reach the variadic-extent ctor
    out.setZero();
    return out;
  } else if constexpr (CEigenDense<B>) {
    if (shape.size() > 2) {
      return fail(errc::rank_mismatch);
    }
    // clang-format off
    const auto rows = static_cast<Eigen::Index>(shape.size() >= 1 ? shape[0] : 1);
    const auto cols = static_cast<Eigen::Index>(shape.size() == 2 ? shape[1] : 1);
    // clang-format on
    // A fixed-size Eigen type can only be the right size already.
    if constexpr (B::RowsAtCompileTime != Eigen::Dynamic) {
      if (B::RowsAtCompileTime != rows) {
        return fail(errc::output_mismatch);
      }
    } else if constexpr (B::ColsAtCompileTime != Eigen::Dynamic) {
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

// Fixed by the operand types alone: on the runtime path the subscript is not a
// type and the return type still has to be one.  So the rank is the widest
// operand's, and a higher one is a rank_mismatch the out() form names its way
// out of.
namespace impl {

template <CEigenFamily X>
inline constexpr int eigen_order_of =
    std::remove_cvref_t<X>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;

template <COperand... Ops>
[[nodiscard]] consteval auto result_probe() noexcept {
  using First = std::remove_cvref_t<first_of_t<Ops...>>;
  using T = scalar_of_t<First>;
  constexpr std::size_t kRank = widest_rank_v<Ops...>;
  if constexpr (CEigenFamily<First>) {
    return std::type_identity<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                            eigen_order_of<First>>>{};
  } else if constexpr (CTensorFamily<First>) {
    // A Tensor's type cannot be rebuilt without naming Eigen's Tensor header.
    return std::type_identity<widest_of_t<Ops...>>{};
  } else if constexpr (CViewFamily<First>) {
    // Contiguous and layout_right, so the executor writes into .data() direct.
    // StaticEinsum overrides this with a static-extent mdarray.
    return std::type_identity<std::experimental::mdarray<
        T, std::dextents<std::size_t, kRank>, std::layout_right>>{};
  } else if constexpr (CNestFamily<First> &&
                       rank_v<widest_of_t<Ops...>> == kRank) {
    // A nest already the right depth answers its own type.
    return std::type_identity<widest_of_t<Ops...>>{};
  } else {
    // A nest shallower than the output answers the vector nest of that rank.
    return std::type_identity<nest_of_t<T, kRank>>{};
  }
}

} // namespace impl

template <COperand... Ops>
  requires CSameFamily<Ops...>
using result_of_t = typename decltype(impl::result_probe<Ops...>())::type;

} // namespace einsum
