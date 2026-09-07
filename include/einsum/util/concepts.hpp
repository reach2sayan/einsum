#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

namespace einsum {

// What a kernel here can multiply and accumulate.  Not std::floating_point:
// the whole library works over int, and over anything else that answers * and
// + the way Eigen's coefficient loops expect.
template <typename T>
concept CScalar =
    std::default_initializable<T> && std::copyable<T> &&
    requires(const T &a, const T &b) {
      { a *b } -> std::convertible_to<T>;
      { a + b } -> std::convertible_to<T>;
    };

// Any range of integrals will do for a shape: a braced list, a Shape, a span,
// an array.  Nothing reads it more than once, so an input_range is enough.
template <typename E>
concept CExtents = std::ranges::input_range<E> && std::integral<std::ranges::range_value_t<E>>;

static_assert(CScalar<int>);
static_assert(CScalar<double>);
static_assert(!CScalar<void *>);

} // namespace einsum
