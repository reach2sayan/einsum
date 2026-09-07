#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/fixed_string.hpp"
#include "einsum/util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <span>
#include <string_view>

namespace einsum {

// What a subscript says, before any operand has been looked at.  The same
// struct comes out of the constexpr parser below and out of the Boost.Parser
// one in src/rt/parse.cpp; tests/tests_parse.cpp checks that they agree.
// Where a term's '...' sits among its written labels, or kNoEllipsis when it
// has none.  A position rather than a flag, because "i...j" puts the broadcast
// axes between the two named ones.
inline constexpr std::uint8_t kNoEllipsis = 255;

struct Subscripts {
  impl::FixedVec<Labels, kMaxOperands> operands{};
  impl::FixedVec<std::uint8_t, kMaxOperands> ellipsis_at{};
  Labels output{};
  std::uint8_t output_ellipsis_at = kNoEllipsis;
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
    // NumPy's rule: the broadcast axes come first, then the labels seen exactly
    // once.  The count is not known until the operands arrive, so the output
    // records only that they lead.
    if (std::ranges::any_of(subs.ellipsis_at,
                            [](const std::uint8_t at) { return at != kNoEllipsis; })) {
      subs.output_ellipsis_at = 0;
    }
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
  std::uint8_t current_ellipsis = kNoEllipsis;
  bool in_output = false;

  // Closing a term keeps its labels and its '...' position together, which is
  // the only reason the two vectors stay the same length.
  const auto close_term = [&]() noexcept {
    const bool ok = subs.operands.try_push_back(current) &&
                    subs.ellipsis_at.try_push_back(current_ellipsis);
    current.clear();
    current_ellipsis = kNoEllipsis;
    return ok;
  };

  for (std::size_t i = 0; i < source.size(); ++i) {
    const char c = source[i];
    if (c == ' ' || c == '\t') {
      continue;
    }
    if (c == '-') {
      if (in_output || i + 1 >= source.size() || source[i + 1] != '>') {
        return fail(errc::bad_syntax);
      }
      if (current.empty() && current_ellipsis == kNoEllipsis) {
        return fail(errc::empty_operand);
      }
      if (!close_term()) {
        return fail(errc::too_many_operands);
      }
      in_output = true;
      subs.explicit_output = true;
      ++i; // the '>' of the arrow
      continue;
    }
    if (c == ',') {
      if (in_output) {
        return fail(errc::bad_syntax);
      }
      if (current.empty() && current_ellipsis == kNoEllipsis) {
        return fail(errc::empty_operand);
      }
      if (!close_term()) {
        return fail(errc::too_many_operands);
      }
      continue;
    }
    if (c == '.') {
      // Exactly three, and at most one per term.  Anything else is a typo, not
      // a shorthand.
      if (i + 2 >= source.size() || source[i + 1] != '.' || source[i + 2] != '.') {
        return fail(errc::bad_syntax);
      }
      i += 2;
      std::uint8_t &at = in_output ? subs.output_ellipsis_at : current_ellipsis;
      if (at != kNoEllipsis) {
        return fail(errc::ellipsis_repeated);
      }
      at = static_cast<std::uint8_t>(in_output ? subs.output.size() : current.size());
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
    if (current.empty() && current_ellipsis == kNoEllipsis) {
      return fail(errc::empty_operand); // a trailing comma
    }
    if (!close_term()) {
      return fail(errc::too_many_operands);
    }
  }
  return finish_subscripts(subs);
}

// --- expanding '...' ----------------------------------------------------------
// Every '...' becomes as many synthetic labels as the operand ranks say it
// stands for, after which nothing downstream knows an ellipsis was ever there.
// The dimensions are right-aligned across operands, as NumPy aligns them: an
// operand covering fewer of them takes the LAST of the broadcast labels, so a
// (3, 4) and a (5, 3, 4) meet on their trailing axes.
[[nodiscard]] constexpr result<Subscripts>
expand(const Subscripts &subs, const std::span<const std::uint8_t> ranks) noexcept {
  if (ranks.size() != subs.operands.size()) {
    return fail(errc::operand_count_mismatch);
  }

  // How many axes each '...' stands for, and the widest of them.
  impl::FixedVec<std::uint8_t, kMaxOperands> covered;
  std::size_t widest = 0;
  for (const auto i : std::views::iota(std::size_t{0}, ranks.size())) {
    const std::size_t named = subs.operands[i].size();
    if (ranks[i] < named) {
      return fail(errc::rank_mismatch);
    }
    const std::size_t nb = ranks[i] - named;
    if (subs.ellipsis_at[i] == kNoEllipsis && nb != 0) {
      return fail(errc::rank_mismatch);
    }
    covered.push_back(static_cast<std::uint8_t>(nb));
    widest = nb > widest ? nb : widest;
  }
  if (widest > kMaxRank) {
    return fail(errc::rank_too_high);
  }

  if (widest == 0 && subs.output_ellipsis_at == kNoEllipsis) {
    // Nothing to expand, and nothing that needed naming.
    return subs;
  }
  if (widest > 0 && subs.output_ellipsis_at == kNoEllipsis) {
    return fail(errc::ellipsis_not_in_output);
  }

  // Splice the last `nb` broadcast labels in at the ellipsis, keeping the
  // written labels either side of it in place.
  const auto splice = [&](const Labels &labels, const std::uint8_t at,
                          const std::size_t nb) noexcept -> result<Labels> {
    Labels out;
    const std::size_t first = widest - nb;
    for (const auto k : std::views::iota(std::size_t{0}, labels.size() + nb)) {
      const bool inside = at != kNoEllipsis && k >= at && k < at + nb;
      const char c = inside ? impl::kBroadcastChars[first + (k - at)]
                            : labels[k < at ? k : k - nb];
      if (!out.try_push_back(c)) {
        return fail(errc::rank_too_high);
      }
    }
    return out;
  };

  Subscripts out = subs;
  for (const auto i : std::views::iota(std::size_t{0}, ranks.size())) {
    const auto term = splice(subs.operands[i], subs.ellipsis_at[i], covered[i]);
    if (!term) {
      return std::unexpected{term.error()};
    }
    out.operands[i] = *term;
    out.ellipsis_at[i] = kNoEllipsis;
  }
  const auto output = splice(subs.output, subs.output_ellipsis_at, widest);
  if (!output) {
    return std::unexpected{output.error()};
  }
  out.output = *output;
  out.output_ellipsis_at = kNoEllipsis;
  return out;
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
