#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>

#define EINSUM_FWD(x) std::forward<decltype(x)>(x)

namespace einsum::impl {

template <std::ranges::input_range R>
[[nodiscard]] constexpr std::ranges::range_value_t<R> product(R &&r) noexcept {
  return std::ranges::fold_left(EINSUM_FWD(r), std::ranges::range_value_t<R>{1},
                                std::multiplies<>{});
}

template <std::ranges::forward_range R>
[[nodiscard]] constexpr std::size_t
index_of(R &&r, const std::ranges::range_value_t<R> &value) noexcept {
  return static_cast<std::size_t>(std::ranges::distance(
      std::ranges::begin(r), std::ranges::find(r, value)));
}

} // namespace einsum::impl
