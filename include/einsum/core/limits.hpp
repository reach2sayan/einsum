#pragma once

#include "einsum/util/fixed_vec.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <string_view>

namespace einsum {

// ptrdiff_t rather than size_t: this is Eigen::Index, and a reversed mdspan
// has negative strides.
using index_t = std::ptrdiff_t;

// The width of a std::array in every constexpr structure below, so they are
// what a Plan costs whether or not the subscript uses them.
inline constexpr std::size_t kMaxRank = 8;
inline constexpr std::size_t kMaxOperands = 8;

using Shape = impl::FixedVec<index_t, kMaxRank>;
using Labels = impl::FixedVec<char, kMaxRank>;

namespace impl {

// Every label, in the order an implicit output is in, so walking this makes
// "ba" infer "->ab".  The digits lead and are the synthetic labels an expanded
// '...' turns into, which is what puts NumPy's ellipsis dims first.
inline constexpr std::string_view kBroadcastChars = "01234567";
inline constexpr std::string_view kLabelChars =
    "01234567ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
inline constexpr std::size_t kLabelSlots = kLabelChars.size();

[[nodiscard]] constexpr std::size_t label_slot(const char c) noexcept {
  if (c <= '9') {
    return static_cast<std::size_t>(c - '0');
  }
  return c >= 'a' ? static_cast<std::size_t>(c - 'a') + 34
                  : static_cast<std::size_t>(c - 'A') + 8;
}

// Only a label an expanded '...' produced may broadcast against a mismatched
// extent, and only one prints back as '...'.
[[nodiscard]] constexpr bool is_broadcast_label(const char c) noexcept {
  return c >= '0' && c <= '9';
}

// One value per label, indexed by the label: the parser has already refused
// every character that is not one.
template <std::regular T> struct LabelTable {
  std::array<T, kLabelSlots> slots{};

  [[nodiscard]] constexpr T &operator[](const char c) noexcept {
    return slots[label_slot(c)];
  }
  [[nodiscard]] constexpr const T &operator[](const char c) const noexcept {
    return slots[label_slot(c)];
  }
};

} // namespace impl

} // namespace einsum
