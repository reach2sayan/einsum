#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/fixed_string.hpp"
#include "einsum/util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <string_view>

namespace einsum {

// What a subscript says, before any operand has been looked at.  The same
// struct comes out of the constexpr parser below and out of the Boost.Parser
// one in src/rt/parse.cpp; tests/tests_parse.cpp checks that they agree.
struct Subscripts {
  impl::FixedVec<Labels, kMaxOperands> operands{};
  Labels output{};
  bool explicit_output = false;

  [[nodiscard]] friend constexpr bool
  operator==(const Subscripts &, const Subscripts &) noexcept = default;
};

namespace impl {

[[nodiscard]] constexpr bool is_label(const char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// How often each label appears.  once_per_operand counts a repeat inside one
// operand as one occurrence, which is what "does anybody else have this label"
// means; counting every occurrence is what an implicit output needs. Saturating
// at two, because two occurrences and twenty mean the same thing to both
// callers.
[[nodiscard]] constexpr LabelTable<std::uint8_t>
count_labels(const FixedVec<Labels, kMaxOperands> &operands,
             const bool once_per_operand) noexcept {
  LabelTable<std::uint8_t> counts;
  for (const Labels &op : operands) {
    // enumerate, because "is this the first time this operand says c" is a
    // question about the position as well as the character.
    for (const auto [a, c] : op | std::views::enumerate) {
      if (once_per_operand && index_of(op, c) != static_cast<std::size_t>(a)) {
        continue;
      }
      std::uint8_t &n = counts[c];
      n = static_cast<std::uint8_t>(n < 2 ? n + 1 : n);
    }
  }
  return counts;
}

// NumPy's implicit output: the labels that occur exactly once across the whole
// input, in ascending character order.  Note "sorted", not "first seen" --
// "ba" infers "->ab", which is the one behaviour change from v1.
[[nodiscard]] constexpr Labels
implicit_output(const FixedVec<Labels, kMaxOperands> &operands) noexcept {
  const LabelTable<std::uint8_t> seen = count_labels(operands, false);
  Labels out;
  // kLabelChars, not the table's own slot order: the output is sorted by
  // character, and 'Z' sorts before 'a'.
  auto once = kLabelChars | std::views::filter(
                                [&seen](const char c) { return seen[c] == 1; });
  // Cannot overflow: a label occurring once is one of at most kMaxRank distinct
  // ones per operand.  Checked anyway.
  for (const char c : once) {
    if (!out.try_push_back(c)) {
      return out;
    }
  }
  return out;
}

// What neither parser needs a grammar to see.
[[nodiscard]] constexpr result<void>
precheck(const std::string_view source) noexcept {
  if (std::ranges::contains(source, '.')) {
    return fail(errc::ellipsis_unsupported);
  }
  const auto blank = [](const char c) noexcept {
    return c == ' ' || c == '\t';
  };
  return std::ranges::all_of(source, blank) ? fail(errc::no_operands)
                                            : result<void>{};
}

// An explicit output has to name labels that exist, and name each one once.
[[nodiscard]] constexpr result<void>
validate_output(const Subscripts &subs) noexcept {
  for (const char c : subs.output) {
    const bool known =
        std::ranges::any_of(subs.operands, [c](const Labels &op) {
          return std::ranges::contains(op, c);
        });
    if (!known) {
      return fail(errc::unknown_output_label);
    }
    if (std::ranges::count(subs.output, c) > 1) {
      return fail(errc::repeated_output_label);
    }
  }
  return {};
}

// The last step of either parser: an explicit output is checked, an absent one
// is inferred.
[[nodiscard]] constexpr result<Subscripts>
finish_subscripts(Subscripts subs) noexcept {
  if (subs.operands.empty()) {
    return fail(errc::no_operands);
  }
  if (!subs.explicit_output) {
    subs.output = implicit_output(subs.operands);
    return subs;
  }
  return validate_output(subs).transform([&] noexcept { return subs; });
}

[[nodiscard]] constexpr result<Subscripts>
parse_subscripts(const std::string_view source) noexcept {
  if (const auto ok = precheck(source); !ok) {
    return std::unexpected{ok.error()};
  }

  Subscripts subs;
  Labels current;
  bool in_output = false;

  for (std::size_t i = 0; i < source.size(); ++i) {
    const char c = source[i];
    if (c == ' ' || c == '\t') {
      continue;
    }
    if (c == '-') {
      if (in_output || i + 1 >= source.size() || source[i + 1] != '>') {
        return fail(errc::bad_syntax);
      }
      if (current.empty()) {
        return fail(errc::empty_operand);
      }
      if (!subs.operands.try_push_back(current)) {
        return fail(errc::too_many_operands);
      }
      current.clear();
      in_output = true;
      subs.explicit_output = true;
      ++i; // the '>' of the arrow
      continue;
    }
    if (c == ',') {
      if (in_output) {
        return fail(errc::bad_syntax);
      }
      if (current.empty()) {
        return fail(errc::empty_operand);
      }
      if (!subs.operands.try_push_back(current)) {
        return fail(errc::too_many_operands);
      }
      current.clear();
      continue;
    }
    if (!is_label(c)) {
      return fail(errc::bad_syntax);
    }
    Labels &target = in_output ? subs.output : current;
    if (!target.try_push_back(c)) {
      return fail(errc::rank_too_high);
    }
  }

  if (!in_output) {
    if (current.empty()) {
      return fail(errc::empty_operand); // a trailing comma
    }
    if (!subs.operands.try_push_back(current)) {
      return fail(errc::too_many_operands);
    }
  }
  return finish_subscripts(subs);
}

} // namespace impl

namespace ct {

// The subscript of einsum<S>, and one static_assert per error code so a bad one
// reports the sentence rather than the instantiation.
template <impl::FixedString S> struct subscripts {
  static constexpr auto parsed = impl::parse_subscripts(S.view());

#define EINSUM_SUBSCRIPT_FAILED(code) failed_with(parsed, errc::code)
  EINSUM_ASSERT_NO_ERROR(EINSUM_SUBSCRIPT_FAILED)
#undef EINSUM_SUBSCRIPT_FAILED

  static constexpr Subscripts value = parsed.value_or(Subscripts{});
};

} // namespace ct

} // namespace einsum
