#pragma once

#include <concepts>

namespace einsum {

// What a kernel here can multiply and accumulate.  Not std::floating_point:
// the library works over int too, and over anything answering * and + the way
// Eigen's coefficient loops expect.
template <typename T>
concept CScalar = std::default_initializable<T> && std::copyable<T> &&
                  requires(const T &a, const T &b) {
                    { a *b } -> std::convertible_to<T>;
                    { a + b } -> std::convertible_to<T>;
                  };

static_assert(CScalar<int>);
static_assert(CScalar<double>);
static_assert(!CScalar<void *>);

} // namespace einsum
