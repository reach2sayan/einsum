#pragma once

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <ostream>
#include <string_view>

namespace einsum {

// The one table.  It generates the enumerators, the message strings, and -- in
// ct/subscripts.hpp -- one static_assert per code, so a subscript rejected at
// compile time reports the same sentence a runtime parse would have returned.
// A Boost.PP sequence of (name, message) tuples rather than an X-macro, because
// the compile-time side has to iterate it inside a class body.
//
// clang-format off
#define EINSUM_ERRC_SEQ                                                                          \
  /* The subscript itself. */                                                                    \
  ((bad_syntax,             "the subscript has a character this grammar does not accept"))       \
  ((ellipsis_unsupported,   "'...' is not supported: name every axis"))                          \
  ((empty_operand,          "an operand between the commas has no labels"))                      \
  ((no_operands,            "the subscript names no operands"))                                  \
  ((too_many_operands,      "more operands than einsum::kMaxOperands"))                          \
  ((rank_too_high,          "an operand has more labels than einsum::kMaxRank"))                 \
  ((unknown_output_label,   "the output names a label that no operand has"))                     \
  ((repeated_output_label,  "the output repeats a label"))                                       \
                                                                                                 \
  /* The subscript against the operands it was handed. */                                        \
  ((operand_count_mismatch, "the subscript and the call disagree on how many operands there are")) \
  ((rank_mismatch,          "an operand's rank differs from the number of labels it was given")) \
  ((extent_conflict,        "one label is bound to two different extents"))                      \
  ((output_mismatch,        "the output's rank or extents are not the ones the subscript implies")) \
  ((size_mismatch,          "the range is smaller than the shape it was given"))                 \
  ((workspace_too_small,    "the borrowed workspace is smaller than the plan needs"))            \
  ((not_matrix,             "to_matrix() needs a result of rank 2 or less"))
// clang-format on

enum class errc : std::uint8_t {
#define EINSUM_ERRC_ENUMERATOR(r, unused, elem) BOOST_PP_TUPLE_ELEM(0, elem),
  BOOST_PP_SEQ_FOR_EACH(EINSUM_ERRC_ENUMERATOR, ~, EINSUM_ERRC_SEQ)
#undef EINSUM_ERRC_ENUMERATOR
};

// No message string and no source location: an error travels the numeric path,
// where an allocation is as unwelcome as the throw it replaces.  The text sits
// in a static table the formatter reads.
struct error {
  errc code;
  [[nodiscard]] friend constexpr bool operator==(error, error) noexcept = default;
};

namespace detail {

inline constexpr std::array kMessages{
#define EINSUM_ERRC_MESSAGE(r, unused, elem)                                   \
  std::string_view{BOOST_PP_TUPLE_ELEM(1, elem)},
    BOOST_PP_SEQ_FOR_EACH(EINSUM_ERRC_MESSAGE, ~, EINSUM_ERRC_SEQ)
#undef EINSUM_ERRC_MESSAGE
};

[[nodiscard]] constexpr std::string_view message(const errc c) noexcept {
  const auto i = static_cast<std::size_t>(c);
  return i < kMessages.size() ? kMessages[i] : "?";
}

} // namespace detail

using detail::message;

inline std::ostream &operator<<(std::ostream &out, const errc c) {
  return out << detail::message(c);
}

inline std::ostream &operator<<(std::ostream &out, const error e) {
  return out << e.code;
}

template <typename T> using result = std::expected<T, error>;

[[nodiscard]] constexpr std::unexpected<error> fail(const errc c) noexcept {
  return std::unexpected{error{.code = c}};
}

// Which failure, if it was one.  A function rather than a variable template
// over the expected itself, because std::expected is not a structural type and
// so cannot be a template argument.
template <typename T>
[[nodiscard]] constexpr bool failed_with(const result<T> &r, const errc c) noexcept {
  return !r.has_value() && r.error().code == c;
}

} // namespace einsum

// One static_assert per error code, generated from the same table the codes
// are: a compile-time einsum reports the sentence a runtime one would have
// returned, and a code added to EINSUM_ERRC_SEQ is diagnosed here for free.
//
// `predicate` is a function-like macro taking an enumerator name and yielding a
// constant expression that is true when that is what went wrong.
#define EINSUM_ERRC_ASSERT_ONE(r, predicate, elem)                               static_assert(!predicate(BOOST_PP_TUPLE_ELEM(0, elem)),                                      "einsum<\"...\">: " BOOST_PP_TUPLE_ELEM(1, elem));
#define EINSUM_ASSERT_NO_ERROR(predicate)                                        BOOST_PP_SEQ_FOR_EACH(EINSUM_ERRC_ASSERT_ONE, predicate, EINSUM_ERRC_SEQ)

// Deriving from the string_view formatter, so a caller's "{:>16}" reaches the
// text.
template <>
struct std::formatter<einsum::errc, char> : std::formatter<std::string_view, char> {
  auto format(const einsum::errc c, std::format_context &ctx) const {
    return std::formatter<std::string_view, char>::format(
        einsum::detail::message(c), ctx);
  }
};

template <>
struct std::formatter<einsum::error, char> : std::formatter<einsum::errc, char> {
  auto format(const einsum::error e, std::format_context &ctx) const {
    return std::formatter<einsum::errc, char>::format(e.code, ctx);
  }
};
