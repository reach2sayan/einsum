#pragma once

#include "einsum/util/fixed_vec.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace einsum {

// ptrdiff_t rather than size_t: this is Eigen::Index, and every stride here is
// signed because a reversed mdspan has negative ones.
using index_t = std::ptrdiff_t;

// Both are the width of a std::array in every constexpr structure below, so
// they are what a Plan costs whether or not the subscript uses them.  Eight
// covers every einsum anyone writes by hand; past that the errc says so.
inline constexpr std::size_t kMaxRank = 8;
inline constexpr std::size_t kMaxOperands = 8;

using Shape = impl::FixedVec<index_t, kMaxRank>;
using Labels = impl::FixedVec<char, kMaxRank>;

namespace impl {

// Every label there is, in ascending character order -- which is also the order
// an implicit output is in, so iterating this is what makes "ba" infer "->ab".
inline constexpr std::string_view kLabelChars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
inline constexpr std::size_t kLabelSlots = kLabelChars.size();

[[nodiscard]] constexpr std::size_t label_slot(const char c) noexcept {
  return c >= 'a' ? static_cast<std::size_t>(c - 'a') + 26
                  : static_cast<std::size_t>(c - 'A');
}

// One value per label, indexed by the label itself.  A dense table beats a
// search, and the parser has already refused every character that is not one.
// Three things want one -- how many operands mention a label, how often it
// occurs, and what extent it is bound to -- and this is all three of them.
template <typename T> struct LabelTable {
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
