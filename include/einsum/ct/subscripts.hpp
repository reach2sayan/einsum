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

// Where a term's '...' sits among its written labels, or this when it has none.
inline constexpr std::uint8_t kNoEllipsis = 255;

// What a subscript says, before any operand has been looked at.  The same
// struct comes out of the constexpr parser below and out of the Boost.Parser
// one in src/rt/parse.cpp; tests/tests_parse.cpp checks that they agree.
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

// How often each label appears, saturating at two.  once_per_operand counts a
// repeat inside one operand once, which is what "does anybody else have this
// label" means; every occurrence is what an implicit output needs.
[[nodiscard]] constexpr LabelTable<std::uint8_t>
count_labels(const FixedVec<Labels, kMaxOperands> &operands,
             const bool once_per_operand) noexcept {
  LabelTable<std::uint8_t> counts;
  for (const Labels &op : operands) {
    const auto first_here = [&](const auto &pair) noexcept {
      const auto [at, c] = pair;
      return !once_per_operand || index_of(op, c) == static_cast<std::size_t>(at);
    };
    for (const char c : op | std::views::enumerate |
                            std::views::filter(first_here) | std::views::values) {
      std::uint8_t &n = counts[c];
      n = static_cast<std::uint8_t>(n < 2 ? n + 1 : n);
    }
  }
  return counts;
}

// NumPy's implicit output: the labels occurring exactly once across the whole
// input, in ascending character order -- so "ba" infers "->ab".  kLabelChars,
// not the table's slot order, because 'Z' sorts before 'a'.  A longer run than
// kMaxRank cannot happen and would truncate.
[[nodiscard]] constexpr Labels
implicit_output(const FixedVec<Labels, kMaxOperands> &operands) noexcept {
  const LabelTable<std::uint8_t> seen = count_labels(operands, false);
  return {std::from_range,
          kLabelChars | std::views::filter(
                            [&seen](const char c) { return seen[c] == 1; })};
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
  const auto known = [&](const char c) noexcept {
    return std::ranges::any_of(subs.operands, [c](const Labels &op) {
      return std::ranges::contains(op, c);
    });
  };
  if (!std::ranges::all_of(subs.output, known)) {
    return fail(errc::unknown_output_label);
  } else if (std::ranges::any_of(subs.output, [&](const char c) {
               return std::ranges::count(subs.output, c) > 1;
             })) {
    return fail(errc::repeated_output_label);
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
    // NumPy's rule: the broadcast axes lead, then the once-labels.  How many
    // there are is not known until the operands arrive.
    if (std::ranges::any_of(subs.ellipsis_at, [](const std::uint8_t at) {
          return at != kNoEllipsis;
        })) {
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

  // A term's labels and its '...' position are stored together, which is what
  // keeps the two vectors the same length.
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
      // Exactly three, and at most one per term.
      if (i + 2 >= source.size() || source[i + 1] != '.' ||
          source[i + 2] != '.') {
        return fail(errc::bad_syntax);
      }
      i += 2;
      std::uint8_t &at = in_output ? subs.output_ellipsis_at : current_ellipsis;
      if (at != kNoEllipsis) {
        return fail(errc::ellipsis_repeated);
      }
      at = static_cast<std::uint8_t>(in_output ? subs.output.size()
                                               : current.size());
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

// Every '...' becomes as many synthetic labels as the operand ranks say it
// stands for, after which nothing downstream knows an ellipsis was there.  The
// dimensions are right-aligned as NumPy aligns them: an operand covering fewer
// of them takes the LAST of the broadcast labels, so (3, 4) and (5, 3, 4) meet
// on their trailing axes.
[[nodiscard]] constexpr result<Subscripts>
expand(const Subscripts &subs,
       const std::span<const std::uint8_t> ranks) noexcept {
  if (ranks.size() != subs.operands.size()) {
    return fail(errc::operand_count_mismatch);
  }

  FixedVec<std::uint8_t, kMaxOperands> covered;
  for (const auto &[rank, labels, at] :
       std::views::zip(ranks, subs.operands, subs.ellipsis_at)) {
    if (rank < labels.size()) {
      return fail(errc::rank_mismatch);
    }
    const std::size_t nb = rank - labels.size();
    if (at == kNoEllipsis && nb != 0) {
      return fail(errc::rank_mismatch);
    }
    covered.push_back(static_cast<std::uint8_t>(nb));
  }
  const std::size_t widest =
      covered.empty() ? 0 : std::ranges::max(covered);
  if (widest > kMaxRank) {
    return fail(errc::rank_too_high);
  }

  if (widest == 0 && subs.output_ellipsis_at == kNoEllipsis) {
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
  for (auto &&[term, at, in_labels, in_at, nb] :
       std::views::zip(out.operands, out.ellipsis_at, subs.operands,
                       subs.ellipsis_at, covered)) {
    const auto spliced = splice(in_labels, in_at, nb);
    if (!spliced) {
      return std::unexpected{spliced.error()};
    }
    term = *spliced;
    at = kNoEllipsis;
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
