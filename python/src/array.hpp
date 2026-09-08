#pragma once

#include "error.hpp"

#include "einsum/core/limits.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/error.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <boost/container/static_vector.hpp>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <string>
#include <vector>

// NumPy arrays as einsum operands.  A Layout is extents and signed strides in
// elements, which is what a NumPy array is, so a sliced, transposed or reversed
// array goes to the kernels where it lies; only a foreign dtype is copied.
namespace einsum::py {

namespace pyb = pybind11;

// The two scalars Eigen's GEMM runs; anything else is converted to double, as
// NumPy's own promotion would.
enum class dtype : bool { f64, f32 };

// Held for the length of a call: the array pybind11 owns and the view into
// it.
template <typename T> struct Operand {
  pyb::array_t<T> array;
  impl::TensorView<const T> view;
};

// float32 throughout is a float32 call; anything else is a double one.
[[nodiscard]] inline dtype promote(const pyb::args &operands) {
  const auto f32 = pyb::dtype::of<float>();
  return std::ranges::all_of(operands,
                             [&f32](const pyb::handle op) {
                               return pyb::isinstance<pyb::array>(op) &&
                                      pyb::reinterpret_borrow<pyb::array>(op)
                                          .dtype()
                                          .is(f32);
                             })
             ? dtype::f32
             : dtype::f64;
}

// An axis of extent > 1 with stride 0 is a broadcast view (np.broadcast_to),
// not a strided rectangle: a repeated element nests with nothing.  Only such an
// operand is materialised.
template <typename T>
[[nodiscard]] bool needs_packing(const pyb::array_t<T> &array) noexcept {
  const auto axes = std::views::iota(pyb::ssize_t{0}, array.ndim());
  return std::ranges::any_of(axes, [&array](const pyb::ssize_t axis) {
    return array.strides(axis) == 0 && array.shape(axis) > 1;
  });
}

// forcecast without c_style: an array whose dtype already matches must not be
// copied, and its strides are what the call uses.
template <typename T>
[[nodiscard]] Operand<T> as_operand(const pyb::handle source) {
  auto array = pyb::array_t<T, pyb::array::forcecast>::ensure(source);
  if (!array) {
    throw pyb::type_error(
        "einsum: every operand must be an array or something array() accepts");
  }
  if (needs_packing(array)) {
    array = pyb::array_t<T, pyb::array::c_style | pyb::array::forcecast>::ensure(
        pyb::module_::import("numpy").attr("ascontiguousarray")(array));
  }
  if (static_cast<std::size_t>(array.ndim()) > kMaxRank) {
    fail_with(errc::rank_too_high);
  }

  const auto itemsize = array.itemsize();
  const auto axes = std::views::iota(pyb::ssize_t{0}, array.ndim());
  // NumPy counts strides in bytes and einsum counts them in elements.
  const impl::Layout layout{
      .shape = {std::from_range,
                axes | std::views::transform([&](const pyb::ssize_t axis) {
                  return static_cast<index_t>(array.shape(axis));
                })},
      .strides = {std::from_range,
                  axes | std::views::transform([&](const pyb::ssize_t axis) {
                    return static_cast<index_t>(array.strides(axis) / itemsize);
                  })}};
  const T *const data = array.data();
  return {std::move(array), impl::TensorView<const T>{data, layout}};
}

// At the shape the lowering chose -- the rank the subscript implies, not any
// operand's -- which is why "i,j->ij" is an array here and not an error.
template <typename T> [[nodiscard]] pyb::array_t<T> empty_like(const Shape &shape) {
  const std::vector<pyb::ssize_t> extents{
      std::from_range,
      shape | std::views::transform(
                  [](const index_t e) { return static_cast<pyb::ssize_t>(e); })};
  return pyb::array_t<T>{extents};
}

} // namespace einsum::py
