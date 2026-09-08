#pragma once

#include "einsum/util/error.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <type_traits>

// libeinsum is compiled -fno-exceptions and every refusal travels as a
// result<T>.  This TU cannot be: pybind11's own error_already_set is a C++
// exception, and every conversion below can raise one.  No throw crosses a
// -fno-exceptions frame -- the library never calls back into Python, so the
// only frames a throw here unwinds are this file's.
namespace einsum::py {

// Inherits einsum::error rather than restating it: the code *is* that type, and
// the message is whatever its own formatter says, so a code added to
// EINSUM_ERRC_SEQ needs nothing here.
struct PyError : error, std::runtime_error {
  explicit PyError(const error e)
      : error{e}, std::runtime_error{std::format("{}", e)} {}
  explicit PyError(const errc c) : PyError{error{.code = c}} {}
};

[[noreturn]] inline void fail_with(const errc c) { throw PyError{c}; }

inline void unwrap(const result<void> &r) {
  if (not r) {
    throw PyError{r.error()};
  }
}

// Constrained away from void: a result<void> prvalue would bind here in
// preference to the overload above.
template <typename T>
  requires(!std::is_void_v<T>)
[[nodiscard]] T unwrap(result<T> &&r) {
  if (not r) {
    throw PyError{r.error()};
  }
  return std::move(*r);
}

} // namespace einsum::py
