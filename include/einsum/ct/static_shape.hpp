#pragma once

#include "einsum/core/kind.hpp"
#include "einsum/core/limits.hpp"
#include "einsum/core/view.hpp"

#include <Eigen/Core>
#include <experimental/mdarray>
#include <experimental/mdspan>

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

// The operands whose whole Layout is known without an object, which is what
// lets einsum<"ij,jk->ik">() lower the whole call at compile time: the Geometry
// is then a constant and the scratch an array.  Value functions, not traits.
namespace einsum::ct {

namespace impl {

// A std::array nest, which is the one nest whose extents are in its type.
template <typename X>
[[nodiscard]] consteval bool is_static_array_nest() noexcept {
  if constexpr (einsum::CScalar<X>) {
    return true;
  } else if constexpr (requires { std::tuple_size<X>::value; } &&
                       std::ranges::range<X>) {
    return is_static_array_nest<
        std::remove_cvref_t<std::ranges::range_value_t<X>>>();
  } else {
    return false;
  }
}

template <typename X>
consteval void static_array_extents(Shape &shape) noexcept {
  if constexpr (!einsum::CScalar<X>) {
    shape.push_back(static_cast<index_t>(std::tuple_size<X>::value));
    static_array_extents<std::remove_cvref_t<std::ranges::range_value_t<X>>>(
        shape);
  }
}

} // namespace impl

// Does every extent of this operand live in its type?
template <typename Op> [[nodiscard]] consteval bool is_static() noexcept {
  using B = std::remove_cvref_t<Op>;
  if constexpr (einsum::impl::CEigenTensor<B>) {
    return false; // a Tensor's dimensions are run-time values
  } else if constexpr (einsum::impl::CEigenDense<B>) {
    return B::RowsAtCompileTime != Eigen::Dynamic &&
           B::ColsAtCompileTime != Eigen::Dynamic;
  } else if constexpr (einsum::impl::CMdspanLike<B>) {
    return requires { typename B::extents_type; } &&
           (B::extents_type::rank_dynamic() == 0) &&
           (std::same_as<typename B::layout_type, std::layout_right> ||
            std::same_as<typename B::layout_type, std::layout_left>);
  } else if constexpr (einsum::CNestedIndexable<B>) {
    return impl::is_static_array_nest<B>();
  } else {
    return false;
  }
}

template <typename... Ops> [[nodiscard]] consteval bool all_static() noexcept {
  return sizeof...(Ops) > 0 && (is_static<Ops>() && ...);
}

// The operand's Layout, read entirely from its type.
template <typename Op>
  requires(is_static<Op>())
[[nodiscard]] consteval einsum::impl::Layout layout_of() noexcept {
  using B = std::remove_cvref_t<Op>;
  if constexpr (einsum::impl::CEigenDense<B>) {
    const bool row_order =
        (B::RowsAtCompileTime == 1 || B::ColsAtCompileTime == 1) ||
        static_cast<bool>(B::IsRowMajor);
    const Shape shape = (B::RowsAtCompileTime == 1 || B::ColsAtCompileTime == 1)
                            ? Shape{index_t{B::RowsAtCompileTime} *
                                    index_t{B::ColsAtCompileTime}}
                            : Shape{index_t{B::RowsAtCompileTime},
                                    index_t{B::ColsAtCompileTime}};
    return einsum::impl::make_layout(
        shape, row_order ? einsum::impl::row_major : einsum::impl::col_major);
  } else if constexpr (einsum::impl::CMdspanLike<B>) {
    using E = typename B::extents_type;
    const Shape shape{
        std::from_range,
        std::views::iota(std::size_t{0}, E::rank()) |
            std::views::transform([](const std::size_t i) {
              return static_cast<index_t>(E::static_extent(i));
            })};
    return einsum::impl::make_layout(
        shape, std::same_as<typename B::layout_type, std::layout_right>
                   ? einsum::impl::row_major
                   : einsum::impl::col_major);
  } else {
    Shape shape;
    einsum::ct::impl::static_array_extents<B>(shape);
    return einsum::impl::make_layout(shape, einsum::impl::row_major);
  }
}

namespace detail {
// A Shape is structural, so the result's extents are read straight off the one
// the lowering computed.
template <Shape S, std::size_t... I>
[[nodiscard]] constexpr auto extents_from(std::index_sequence<I...>) noexcept
    -> std::extents<std::size_t, static_cast<std::size_t>(S.data_[I])...> {
  return {};
}
} // namespace detail

template <Shape S>
using extents_of_t =
    decltype(detail::extents_from<S>(std::make_index_sequence<S.size_>{}));

static_assert(
    std::same_as<extents_of_t<Shape{2, 3}>, std::extents<std::size_t, 2, 3>>);
static_assert(std::same_as<extents_of_t<Shape{}>, std::extents<std::size_t>>);

// The view family's result when the whole lowering is a constant: every extent
// in the type and a std::array behind it, so the call touches no allocator.
// The runtime form is an mdarray over dextents and a vector, so a caller moving
// between the two paths keeps the same indexing.
template <einsum::CScalar T, Shape S>
using static_mdarray_t = std::experimental::mdarray<
    T, extents_of_t<S>, std::layout_right,
    std::array<T, static_cast<std::size_t>(einsum::impl::product(S)) == 0
                      ? 1
                      : static_cast<std::size_t>(einsum::impl::product(S))>>;

} // namespace einsum::ct
