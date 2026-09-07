#pragma once

#include "einsum/util/concepts.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <span>
#include <type_traits>

namespace einsum::impl {

// A vector with its capacity in the type and no allocator.  Structural, so a
// Shape can be a template argument (ct/labels.hpp turns one into std::extents);
// that is why data_ and size_ are public and there is no invariant to protect.
//
// push_back returns false rather than growing or throwing: every caller here
// checks it and turns a refusal into an errc, which is what keeps the headers
// free of both allocation and exceptions.
template <typename T, std::size_t N> struct FixedVec {
  std::array<T, N> data_{};
  std::size_t size_ = 0;

  constexpr FixedVec() noexcept = default;

  // Shape{2, 3} rather than Shape{{2, 3}, 2}, and Shape{someSpan} rather than a
  // copy loop at every call: this is what lets flat() and into() take one
  // `const Shape &` instead of an overload per way of spelling a shape.
  //
  // A range longer than N stops at N.  Nothing downstream is fooled: a Shape is
  // only ever handed to a Plan, which compares its rank against the subscript's
  // label count and answers rank_mismatch.
  constexpr FixedVec(const std::initializer_list<T> values) noexcept {
    for (const T &value : values) {
      if (!push_back(value)) {
        return;
      }
    }
  }

  template <std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, T> &&
             (!std::same_as<std::remove_cvref_t<R>, FixedVec>)
  constexpr FixedVec(R &&values) noexcept {
    for (auto &&value : values) {
      if (!push_back(static_cast<T>(value))) {
        return;
      }
    }
  }

  [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] constexpr bool full() const noexcept { return size_ == N; }

  [[nodiscard]] constexpr T *begin() noexcept { return data_.data(); }
  [[nodiscard]] constexpr T *end() noexcept { return data_.data() + size_; }
  [[nodiscard]] constexpr const T *begin() const noexcept { return data_.data(); }
  [[nodiscard]] constexpr const T *end() const noexcept { return data_.data() + size_; }
  [[nodiscard]] constexpr const T *data() const noexcept { return data_.data(); }
  [[nodiscard]] constexpr T *data() noexcept { return data_.data(); }

  [[nodiscard]] constexpr T &operator[](const std::size_t i) noexcept { return data_[i]; }
  [[nodiscard]] constexpr const T &operator[](const std::size_t i) const noexcept {
    return data_[i];
  }
  [[nodiscard]] constexpr T &back() noexcept { return data_[size_ - 1]; }
  [[nodiscard]] constexpr const T &back() const noexcept { return data_[size_ - 1]; }

  constexpr bool push_back(const T &value) noexcept {
    if (size_ == N) {
      return false;
    }
    data_[size_++] = value;
    return true;
  }

  constexpr void clear() noexcept { size_ = 0; }

  // Grows with value-initialised elements, shrinks by forgetting, and never
  // past N.  The one caller that wants a length rather than a sequence.
  constexpr void resize(const std::size_t n) noexcept {
    const std::size_t want = n < N ? n : N;
    for (std::size_t i = size_; i < want; ++i) {
      data_[i] = T{};
    }
    size_ = want;
  }

  constexpr void erase_at(const std::size_t i) noexcept {
    std::shift_left(data_.data() + i, data_.data() + size_, 1);
    --size_;
  }

  [[nodiscard]] constexpr std::span<const T> span() const noexcept {
    return {data_.data(), size_};
  }
  [[nodiscard]] constexpr std::span<T> span() noexcept {
    return {data_.data(), size_};
  }

  // Over the live prefix only: two vectors that agree on their elements are
  // equal whatever the tails of their arrays still hold.
  [[nodiscard]] constexpr bool operator==(const FixedVec &other) const noexcept {
    return size_ == other.size_ && std::equal(begin(), end(), other.begin());
  }

  [[nodiscard]] constexpr bool contains(const T &value) const noexcept {
    return std::ranges::find(*this, value) != end();
  }

  // npos-free: the caller compares against size().
  [[nodiscard]] constexpr std::size_t index_of(const T &value) const noexcept {
    return static_cast<std::size_t>(
        std::ranges::distance(begin(), std::ranges::find(*this, value)));
  }
};

static_assert(std::ranges::contiguous_range<FixedVec<int, 4>>);
static_assert(std::ranges::sized_range<FixedVec<int, 4>>);

} // namespace einsum::impl
