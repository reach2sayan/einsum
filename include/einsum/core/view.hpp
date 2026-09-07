#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <Eigen/Core>
#include <experimental/mdspan>

#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <ranges>
#include <span>
#include <type_traits>

namespace einsum {

// --- layout policies ---------------------------------------------------------
// The two orders as types, so a compile-time operand names one and the runtime
// API's enum is a dispatch to the same two functions.
enum class layout : std::uint8_t { row_major, col_major };
inline constexpr layout row_major = layout::row_major;
inline constexpr layout col_major = layout::col_major;

namespace impl {
// The two orders are one walk over the axes in opposite directions: the stride
// of an axis is the product of the extents inside it, and "inside" is the only
// thing they disagree about.
[[nodiscard]] constexpr Shape packed_strides(const Shape &shape, const bool innermost_last) noexcept {
  Shape out = shape; // same rank, strides about to be overwritten
  index_t step = 1;
  for (const std::size_t n : std::views::iota(std::size_t{0}, shape.size())) {
    const std::size_t i = innermost_last ? shape.size() - 1 - n : n;
    out[i] = step;
    step *= shape[i];
  }
  return out;
}
} // namespace impl

struct RowMajor {
  static constexpr layout order = layout::row_major;
  [[nodiscard]] static constexpr Shape strides(const Shape &shape) noexcept {
    return impl::packed_strides(shape, true);
  }
};

struct ColMajor {
  static constexpr layout order = layout::col_major;
  [[nodiscard]] static constexpr Shape strides(const Shape &shape) noexcept {
    return impl::packed_strides(shape, false);
  }
};

template <typename P>
concept CLayoutPolicy = requires(const Shape &shape) {
  { P::order } -> std::convertible_to<layout>;
  { P::strides(shape) } -> std::same_as<Shape>;
};

// Extents and strides, and nothing about who owns the memory.  Strides are in
// elements and signed: a layout_stride mapping is as much a Layout as a packed
// buffer is.
struct Layout {
  Shape shape{};
  Shape strides{};

  [[nodiscard]] constexpr std::size_t rank() const noexcept { return shape.size(); }
  [[nodiscard]] constexpr index_t size() const noexcept { return impl::product(shape); }
  [[nodiscard]] friend constexpr bool operator==(const Layout &, const Layout &) noexcept = default;
};

template <CLayoutPolicy P>
[[nodiscard]] constexpr Layout make_layout(const Shape &shape) noexcept {
  return {.shape = shape, .strides = P::strides(shape)};
}

[[nodiscard]] constexpr Layout make_layout(const Shape &shape,
                                           const layout order = row_major) noexcept {
  return order == row_major ? make_layout<RowMajor>(shape) : make_layout<ColMajor>(shape);
}

// A pointer and a Layout; non-owning, const-correct through T.
template <typename T> struct TensorView {
  using value_type = std::remove_cv_t<T>;

  T *data = nullptr;
  Layout layout{};

  [[nodiscard]] constexpr std::size_t rank() const noexcept { return layout.rank(); }
  [[nodiscard]] constexpr index_t size() const noexcept { return layout.size(); }

  constexpr operator TensorView<const T>() const noexcept
    requires(!std::is_const_v<T>)
  {
    return {data, layout};
  }
};

// The output tag: a view the caller still owns, and what tells bind() where the
// operands stop.  Never const -- einsum writes through it.
template <typename T> struct Into {
  using value_type = T;
  TensorView<T> view{};
};

namespace impl {
template <typename X> inline constexpr bool is_tensor_view_v = false;
template <typename T> inline constexpr bool is_tensor_view_v<TensorView<T>> = true;
template <typename X> inline constexpr bool is_into_v = false;
template <typename T> inline constexpr bool is_into_v<Into<T>> = true;
template <typename X> inline constexpr bool is_view_result_v = false;
template <typename U> inline constexpr bool is_view_result_v<result<TensorView<U>>> = true;
template <typename X> inline constexpr bool is_into_result_v = false;
template <typename U> inline constexpr bool is_into_result_v<result<Into<U>>> = true;

// Eigen's own idea of a dense object: the memory has to be there.  A Product
// has no data() and is not one.
template <typename D>
concept CEigenDense = std::derived_from<D, Eigen::DenseBase<D>> && requires(const D &d) {
  typename D::Scalar;
  { d.outerStride() } -> std::convertible_to<Eigen::Index>;
  { d.data() } -> std::convertible_to<const typename D::Scalar *>;
};

// A vector is rank 1 however it is stored; a matrix is rank 2 with the row
// stride first, which for a column-major object is the inner one.
template <CEigenDense D> [[nodiscard]] Layout eigen_layout(const D &m) noexcept {
  Layout out;
  const auto inner = static_cast<index_t>(m.innerStride());
  if constexpr (D::IsVectorAtCompileTime) {
    (void)out.shape.push_back(static_cast<index_t>(m.size()));
    (void)out.strides.push_back(inner);
  } else {
    const auto outer = static_cast<index_t>(m.outerStride());
    (void)out.shape.push_back(static_cast<index_t>(m.rows()));
    (void)out.shape.push_back(static_cast<index_t>(m.cols()));
    (void)out.strides.push_back(D::IsRowMajor ? outer : inner);
    (void)out.strides.push_back(D::IsRowMajor ? inner : outer);
  }
  return out;
}
} // namespace impl

// --- the three normalisations ------------------------------------------------
// Everything an operand can be reaches TensorView through exactly one of these.
template <impl::CEigenDense D>
[[nodiscard]] TensorView<const typename D::Scalar> as_view(const D &m) noexcept {
  return {m.data(), impl::eigen_layout(m)};
}

template <impl::CEigenDense D>
  requires(!std::is_const_v<D>)
[[nodiscard]] TensorView<typename D::Scalar> as_view(D &m) noexcept {
  return {m.data(), impl::eigen_layout(std::as_const(m))};
}

// extent(i) and stride(i) are what all three mappings answer, layout_stride
// included -- which is what makes a submdspan an operand here.
template <typename T, typename E, typename L, typename A>
  requires std::same_as<A, std::default_accessor<T>>
[[nodiscard]] constexpr TensorView<T> as_view(const std::mdspan<T, E, L, A> &m) noexcept {
  static_assert(E::rank() <= kMaxRank, "as_view: more extents than einsum::kMaxRank");
  TensorView<T> out{m.data_handle(), {}};
  for (const std::size_t i : std::views::iota(std::size_t{0}, E::rank())) {
    (void)out.layout.shape.push_back(static_cast<index_t>(m.extent(i)));
    (void)out.layout.strides.push_back(static_cast<index_t>(m.stride(i)));
  }
  return out;
}

// Flat memory: the one place a shape is a claim that can be false, so the one
// place the size is checked.  A range and a pointer both arrive here.
//
// `const Shape &`, not a range parameter: Shape converts from a braced list and
// from any CExtents range, so one declaration takes all three spellings where
// four overloads used to.
template <typename T>
[[nodiscard]] constexpr result<TensorView<T>>
as_view(const std::span<T> memory, const Shape &shape,
        const layout order = row_major) noexcept {
  const Layout lay = make_layout(shape, order);
  if (lay.size() > static_cast<index_t>(memory.size())) {
    return fail(errc::size_mismatch);
  }
  return TensorView<T>{memory.data(), lay};
}

// flat(v, {2, 3}) and flat(p, {2, 3}): make the span, then the one form above.
// A pointer's span is sized from the shape, so it passes the same check.
template <typename R>
  requires std::ranges::contiguous_range<R> && std::ranges::sized_range<R>
[[nodiscard]] constexpr auto flat(R &r, const Shape &shape,
                                  const layout order = row_major) noexcept {
  return as_view(std::span{std::ranges::data(r), std::ranges::size(r)}, shape, order);
}

template <typename T>
[[nodiscard]] constexpr result<TensorView<T>>
flat(T *memory, const Shape &shape, const layout order = row_major) noexcept {
  return as_view(std::span<T>{memory, static_cast<std::size_t>(impl::product(shape))}, shape,
                 order);
}

// --- the view an operand normalises to ---------------------------------------
namespace impl {
template <typename X> struct view_of {
  using type = decltype(as_view(std::declval<std::remove_cvref_t<X> &>()));
};
template <typename U> struct view_of<TensorView<U>> { using type = TensorView<U>; };
template <typename U> struct view_of<result<TensorView<U>>> { using type = TensorView<U>; };

template <typename X> using view_of_t = typename view_of<std::remove_cvref_t<X>>::type;
template <typename X>
using scalar_of_t = std::remove_cv_t<std::remove_pointer_t<decltype(view_of_t<X>::data)>>;

// One body for every output: a writable view with a tag on it, the fallible
// ones carrying their error through transform().
template <typename V> [[nodiscard]] constexpr auto tag_output(V viewed) noexcept {
  if constexpr (is_view_result_v<V>) {
    using View = typename V::value_type;
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(View::data)>>,
                  "into(): the output is const, and einsum writes through it");
    return viewed.transform([](const View &v) noexcept { return Into<typename View::value_type>{v}; });
  } else {
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(V::data)>>,
                  "into(): the output is const, and einsum writes through it");
    return Into<typename V::value_type>{viewed};
  }
}
} // namespace impl

// --- the output form ---------------------------------------------------------
template <typename X>
  requires requires(X &&x) { as_view(EINSUM_FWD(x)); }
[[nodiscard]] constexpr auto into(X &&x) noexcept {
  return impl::tag_output(as_view(EINSUM_FWD(x)));
}

template <typename X>
[[nodiscard]] constexpr auto into(X &&x, const Shape &shape,
                                  const layout order = row_major) noexcept {
  return impl::tag_output(flat(EINSUM_FWD(x), shape, order));
}

template <typename X>
concept CInto = impl::is_into_v<std::remove_cvref_t<X>> ||
                impl::is_into_result_v<std::remove_cvref_t<X>>;

} // namespace einsum
