#pragma once

#include "einsum/core/kind.hpp"
#include "einsum/core/limits.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <Eigen/Core>
#include <experimental/mdspan>

#include <array>
#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace einsum::impl {

// --- layout policies ---------------------------------------------------------
// The two orders as types, so a compile-time operand names one and the runtime
// API's enum is a dispatch to the same two functions.
enum class layout : std::uint8_t { row_major, col_major };
inline constexpr layout row_major = layout::row_major;
inline constexpr layout col_major = layout::col_major;

// The two orders are one walk over the axes in opposite directions: the stride
// of an axis is the product of the extents inside it, and "inside" is the only
// thing they disagree about.
[[nodiscard]] constexpr Shape
packed_strides(const Shape &shape, const bool innermost_last) noexcept {
  Shape out = shape; // same rank, strides about to be overwritten
  index_t step = 1;
  for (const std::size_t n : std::views::iota(std::size_t{0}, shape.size())) {
    const std::size_t i = innermost_last ? shape.size() - 1 - n : n;
    out[i] = step;
    step *= shape[i];
  }
  return out;
}

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

  [[nodiscard]] constexpr std::size_t rank() const noexcept {
    return shape.size();
  }
  [[nodiscard]] constexpr index_t size() const noexcept {
    return impl::product(shape);
  }
  [[nodiscard]] friend constexpr bool
  operator==(const Layout &, const Layout &) noexcept = default;
};

template <CLayoutPolicy P>
[[nodiscard]] constexpr Layout make_layout(const Shape &shape) noexcept {
  return {.shape = shape, .strides = P::strides(shape)};
}

// --- walking one
// -------------------------------------------------------------- Every offset a
// row-major walk over these layouts visits, in the order it visits them.  The
// layouts name the same axes of different tensors -- the GEMM's three batch
// groups, a permuting copy's two sides -- so they share extents, and one carry
// serves all of them.
//
// The offsets are carried, not recomputed: each axis that wraps gives back
// exactly what it has added since it last did, so a step costs one add per
// operand.  A std::views::cartesian_product of iota views reads better and is
// what this replaced, but recomputing the dot product per element cost +82% on
// a 64x64 transpose and +45% on a batched matmul -- not a price a walk this hot
// can pay.
// One index_t per layout: the callback is handed the operands' offsets, so its
// arity is the operand count and is checked here rather than inside std::apply.
template <typename> using offset_arg_t = index_t;

template <typename F, typename... L>
  requires(sizeof...(L) >= 1) &&
          (std::same_as<std::remove_cvref_t<L>, Layout> && ...) &&
          std::invocable<F &, offset_arg_t<L>...>
constexpr void for_each_offset(F &&fn, const L &...layouts) noexcept {
  constexpr std::size_t kOperands = sizeof...(L);

  // Extents and strides copied into this frame, and the per-operand steps
  // written as a fold rather than a loop.  Reading them through the Layouts
  // instead costs a pointer load and a value load per operand per step, and
  // leaves the operand count a run-time bound the compiler will not unroll --
  // which is most of what the class this replaced was doing better.
  const std::array<std::array<index_t, kMaxRank>, kOperands> strides{
      [](const Layout &one) {
        std::array<index_t, kMaxRank> row{};
        std::ranges::copy(one.strides, row.begin());
        return row;
      }(layouts)...};

  const Shape &shape = std::get<0>(std::tie(layouts...)).shape;
  const std::size_t rank = shape.size();
  std::array<index_t, kMaxRank> extent{};
  std::ranges::copy(shape, extent.begin());

  std::array<index_t, kMaxRank> at{};
  std::array<index_t, kOperands> off{};
  const auto step = [&]<std::size_t... K>(const std::size_t axis,
                                          const index_t by,
                                          std::index_sequence<K...>) noexcept {
    ((off[K] += strides[K][axis] * by), ...);
  };
  for (index_t n = impl::product(shape); n-- > 0;) {
    std::apply(fn, off);
    for (std::size_t axis = rank; axis-- > 0;) {
      step(axis, 1, std::make_index_sequence<kOperands>{});
      if (++at[axis] < extent[axis]) {
        break;
      }
      // The axis wrapped: give back exactly what it has added since it last
      // did.
      step(axis, -extent[axis], std::make_index_sequence<kOperands>{});
      at[axis] = 0;
    }
  }
}

[[nodiscard]] constexpr Layout
make_layout(const Shape &shape, const layout order = row_major) noexcept {
  return order == row_major ? make_layout<RowMajor>(shape)
                            : make_layout<ColMajor>(shape);
}

// A Tensor's own strides: it is densely packed, and its Layout says which end
// moves fastest.  ColMajor here means the FIRST index is fastest, which is
// einsum's col_major.
template <CEigenTensor B>
[[nodiscard]] constexpr Layout tensor_layout(const Shape &shape) noexcept {
  // Both sides through int: Tensor's Layout is its own unnamed enumeration, and
  // comparing two unrelated enumerations is a warning this build treats as an
  // error.
  constexpr bool row_order =
      static_cast<int>(B::Layout) == static_cast<int>(Eigen::RowMajor);
  return make_layout(shape, row_order ? row_major : col_major);
}

// A pointer and a Layout; non-owning, const-correct through T.
template <typename T>
  requires CScalar<std::remove_cv_t<T>>
struct TensorView {
  using value_type = std::remove_cv_t<T>;

  T *data = nullptr;
  Layout layout{};

  [[nodiscard]] constexpr std::size_t rank() const noexcept {
    return layout.rank();
  }
  [[nodiscard]] constexpr index_t size() const noexcept {
    return layout.size();
  }

  constexpr operator TensorView<const T>() const noexcept
    requires(!std::is_const_v<T>)
  {
    return {data, layout};
  }
};

template <typename X> inline constexpr bool is_tensor_view_v = false;
template <typename T>
inline constexpr bool is_tensor_view_v<TensorView<T>> = true;
template <typename X> inline constexpr bool is_view_result_v = false;
template <typename U>
inline constexpr bool is_view_result_v<result<TensorView<U>>> = true;

// A vector is rank 1 however it is stored; a matrix is rank 2 with the row
// stride first, which for a column-major object is the inner one.
template <CEigenDense D>
[[nodiscard]] Layout eigen_layout(const D &m) noexcept {
  Layout out;
  const auto inner = static_cast<index_t>(m.innerStride());
  if constexpr (D::IsVectorAtCompileTime) {
    out.shape.push_back(static_cast<index_t>(m.size()));
    out.strides.push_back(inner);
  } else {
    const auto outer = static_cast<index_t>(m.outerStride());
    out.shape.push_back(static_cast<index_t>(m.rows()));
    out.shape.push_back(static_cast<index_t>(m.cols()));
    out.strides.push_back(D::IsRowMajor ? outer : inner);
    out.strides.push_back(D::IsRowMajor ? inner : outer);
  }
  return out;
}

// --- the three normalisations ------------------------------------------------
// Everything an operand can be reaches TensorView through exactly one of these.
template <impl::CEigenDense D>
[[nodiscard]] TensorView<const typename D::Scalar>
as_view(const D &m) noexcept {
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
[[nodiscard]] constexpr TensorView<T>
as_view(const std::mdspan<T, E, L, A> &m) noexcept {
  static_assert(E::rank() <= kMaxRank,
                "as_view: more extents than einsum::kMaxRank");
  TensorView<T> out{m.data_handle(), {}};
  for (const std::size_t i : std::views::iota(std::size_t{0}, E::rank())) {
    out.layout.shape.push_back(static_cast<index_t>(m.extent(i)));
    out.layout.strides.push_back(static_cast<index_t>(m.stride(i)));
  }
  return out;
}

// --- reading an operand ------------------------------------------------------
// Extents first, because everything else needs them.  A nest is measured by
// descending its first element, and every sibling has to agree: a ragged nest
// is not a tensor, and says so rather than reading past a short row.

// Depth D of R, measuring as it descends.  The first node at each level sets
// that level's extent and every later one is held to it, so a ragged nest is
// caught where it differs rather than read past.
template <std::size_t D, std::size_t R, typename X>
[[nodiscard]] constexpr result<void>
measure_nest(const X &x, std::array<index_t, R> &ext,
             std::array<bool, R> &seen) noexcept {
  if constexpr (D == R) {
    return {};
  } else {
    const auto n = static_cast<index_t>(std::ranges::size(x));
    if (!seen[D]) {
      ext[D] = n;
      seen[D] = true;
    } else if (ext[D] != n) {
      return fail(errc::extent_conflict);
    }
    for (const auto &row : x) {
      if (const auto ok = measure_nest<D + 1, R>(row, ext, seen); !ok) {
        return ok;
      }
    }
    return {};
  }
}

template <COperand X>
[[nodiscard]] constexpr result<Shape> shape_of(const X &x) noexcept {
  using B = std::remove_cvref_t<X>;
  Shape shape;
  if constexpr (impl::CEigenTensor<B>) {
    for (const auto i : std::views::iota(std::size_t{0}, rank_v<B>)) {
      shape.push_back(static_cast<index_t>(x.dimension(i)));
    }
    return shape;
  } else if constexpr (impl::CEigenDense<B>) {
    if constexpr (B::IsVectorAtCompileTime) {
      shape.push_back(static_cast<index_t>(x.size()));
    } else {
      shape.push_back(static_cast<index_t>(x.rows()));
      shape.push_back(static_cast<index_t>(x.cols()));
    }
    return shape;
  } else if constexpr (CNestedIndexable<B>) {
    // One walk that both measures and checks.
    std::array<index_t, rank_v<B>> ext{};
    std::array<bool, rank_v<B>> seen{};
    if (const auto ok = impl::measure_nest<0, rank_v<B>>(x, ext, seen); !ok) {
      return std::unexpected{ok.error()};
    }
    for (const index_t n : ext) {
      shape.push_back(n);
    }
    return shape;
  } else {
    for (const auto i : std::views::iota(std::size_t{0}, rank_v<B>)) {
      shape.push_back(static_cast<index_t>(x.extent(i)));
    }
    return shape;
  }
}

// One element, whichever way its type spells the accessor.  The rank is in the
// operand's type even when its extents are not, so the index pack is built
// once and the three families differ only in how they consume it.

template <std::size_t D, typename X, std::size_t R>
[[nodiscard]] constexpr decltype(auto)
nest_at(X &&x, const std::array<index_t, R> &at) noexcept {
  if constexpr (D == R) {
    return (x);
  } else {
    return nest_at<D + 1>(EINSUM_FWD(x)[static_cast<std::size_t>(at[D])], at);
  }
}

template <COperand X, std::size_t R, std::size_t... I>
[[nodiscard]] constexpr decltype(auto)
element_at(X &&x, const std::array<index_t, R> &at,
           std::index_sequence<I...>) noexcept {
  using B = std::remove_cvref_t<X>;
  if constexpr (CEigenDense<B> || CEigenTensor<B>) {
    return EINSUM_FWD(x)(at[I]...);
  } else if constexpr (CNestedIndexable<B>) {
    return nest_at<0>(EINSUM_FWD(x), at);
  } else {
    return EINSUM_FWD(x)[at[I]...];
  }
}

template <COperand X>
[[nodiscard]] constexpr decltype(auto)
element_at(X &&x, const std::array<index_t, rank_v<X>> &at) noexcept {
  return impl::element_at(EINSUM_FWD(x), at,
                          std::make_index_sequence<rank_v<X>>{});
}

// --- packing an operand the kernels can address ------------------------------
// An operand whose memory is already a strided rectangle is handed to Eigen as
// it stands; anything else -- a vector of vectors, a proxy, a type that only
// answers its accessor -- is walked once and packed row-major into scratch.
template <typename X>
concept CContiguous =
    impl::CEigenDense<std::remove_cvref_t<X>> || requires(const X &x) {
      { x.data_handle() };
      typename std::remove_cvref_t<X>::layout_type;
    };

// Row-major over the operand's own index space, which is the order the packed
// buffer is in and the order infer_output_shape assumes.
template <COperand X>
constexpr void gather(const X &x, scalar_of_t<X> *dst,
                      const Shape &shape) noexcept {
  std::array<index_t, rank_v<X>> at{};
  for (index_t k = impl::product(shape); k-- > 0;) {
    *dst++ = element_at(x, at);
    for (std::size_t a = rank_v<X>; a-- > 0;) {
      if (++at[a] < shape[a]) {
        break;
      }
      at[a] = 0;
    }
  }
}

// The same walk backwards, for a result whose type only answers its accessor.
template <COperand X>
constexpr void scatter(const scalar_of_t<X> *src, X &x,
                       const Shape &shape) noexcept {
  std::array<index_t, rank_v<X>> at{};
  for (index_t k = impl::product(shape); k-- > 0;) {
    element_at(x, at) = *src++;
    for (std::size_t a = rank_v<X>; a-- > 0;) {
      if (++at[a] < shape[a]) {
        break;
      }
      at[a] = 0;
    }
  }
}

} // namespace einsum::impl
