#pragma once

#include <algorithm>
#include <functional>
#include <ranges>
#include <utility>

#define EINSUM_FWD(x) std::forward<decltype(x)>(x)

namespace einsum::impl {

// The extent of a shape, and 1 for the rank-0 one -- which is the identity
// fold_left wants anyway, so the empty case needs no special mention.
template <std::ranges::input_range R>
[[nodiscard]] constexpr std::ranges::range_value_t<R> product(R &&r) noexcept {
  return std::ranges::fold_left(EINSUM_FWD(r), std::ranges::range_value_t<R>{1},
                                std::multiplies<>{});
}

} // namespace einsum::impl
