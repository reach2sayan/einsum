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

// NumPy arrays as einsum operands.
//
// An impl::Layout is extents and signed strides in elements, which is what a
// NumPy array is -- so a C-ordered, F-ordered, sliced, transposed or reversed
// array is handed to the kernels where it already lies, and only a dtype this
// module does not run in gets copied.  That the strides are signed is why the
// reversed case works at all; the library made them ptrdiff_t for exactly it.
namespace einsum::py {

namespace pyb = pybind11;

// The two scalars the module runs in.  Both go through Eigen's GEMM; anything
// else a caller passes is converted to double on the way in, which is what
// NumPy's own promotion would have done.
enum class dtype : bool { f64, f32 };

// One operand, held for the length of a call: the array pybind11 owns (which is
// the caller's own when nothing had to be converted) and the view into it.
template <typename T> struct Operand {
  pyb::array_t<T> array;
  impl::TensorView<const T> view;
};

// Every operand's dtype, promoted the way NumPy promotes: float32 throughout is
// a float32 call, and anything else is a double one.
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

// An axis of extent > 1 whose stride is 0 is a broadcast view -- np.broadcast_to
// and what an np.newaxis expansion leaves behind.  It is not a strided
// rectangle: collapse() reasons about strides that nest, and a repeated element
// nests with nothing.  Such an operand is materialised, and only such an one.
template <typename T>
[[nodiscard]] bool needs_packing(const pyb::array_t<T> &array) noexcept {
  const auto axes = std::views::iota(pyb::ssize_t{0}, array.ndim());
  return std::ranges::any_of(axes, [&array](const pyb::ssize_t axis) {
    return array.strides(axis) == 0 && array.shape(axis) > 1;
  });
}

// forcecast, and not c_style with it: a cast copies, and an array whose dtype
// already matches must not.  Its strides are what the call is going to use.
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

  impl::Layout layout;
  const auto itemsize = array.itemsize();
  for (const auto axis : std::views::iota(pyb::ssize_t{0}, array.ndim())) {
    layout.shape.push_back(static_cast<index_t>(array.shape(axis)));
    // NumPy counts strides in bytes and einsum counts them in elements.
    layout.strides.push_back(
        static_cast<index_t>(array.strides(axis) / itemsize));
  }
  const T *const data = array.data();
  return {std::move(array), impl::TensorView<const T>{data, layout}};
}

// The result: allocated at the shape the lowering chose, which is the rank the
// *subscript* implies rather than any operand's.  That is the whole of what the
// dynamic entry buys, and it is why "i,j->ij" is an array here and not an error.
template <typename T> [[nodiscard]] pyb::array_t<T> empty_like(const Shape &shape) {
  std::vector<pyb::ssize_t> extents;
  extents.reserve(shape.size());
  std::ranges::transform(shape, std::back_inserter(extents),
                         [](const index_t e) {
                           return static_cast<pyb::ssize_t>(e);
                         });
  return pyb::array_t<T>{extents};
}

} // namespace einsum::py
