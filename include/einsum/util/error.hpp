#pragma once

#include <boost/describe/enum.hpp>
#include <boost/describe/enum_to_string.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/transform.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <ostream>
#include <string_view>

namespace einsum {

// The one table: it generates the enumerators, the message strings, and one
// static_assert per code, so a subscript rejected at compile time reports the
// sentence a runtime parse would have returned.  A Boost.PP sequence rather
// than an X-macro, because the compile-time side iterates it in a class body.
//
// clang-format off
#define EINSUM_ERRC_SEQ                                                                          \
  ((bad_syntax,             "the subscript has a character this grammar does not accept"))       \
  ((ellipsis_repeated,      "a term has more than one '...'"))                                 \
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
  ((broadcast_mismatch,     "the '...' dimensions do not broadcast"))                            \
  ((ellipsis_not_in_output, "the operands' '...' covers axes the output does not name"))         \
  ((output_mismatch,        "the output's rank or extents are not the ones the subscript implies"))
// clang-format on

// The enumerator names on their own, so the enum is generated from the table.
#define EINSUM_ERRC_NAME(s, unused, elem) BOOST_PP_TUPLE_ELEM(0, elem)
#define EINSUM_ERRC_NAMES                                                      \
  BOOST_PP_SEQ_TRANSFORM(EINSUM_ERRC_NAME, ~, EINSUM_ERRC_SEQ)

enum class errc : std::uint8_t { BOOST_PP_SEQ_ENUM(EINSUM_ERRC_NAMES) };

BOOST_DESCRIBE_ENUM(errc, BOOST_PP_SEQ_ENUM(EINSUM_ERRC_NAMES))

struct error {
  errc code;
  [[nodiscard]] friend constexpr bool operator==(error,
                                                 error) noexcept = default;
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

[[nodiscard]] constexpr std::string_view message(const error e) noexcept {
  return message(e.code);
}

} // namespace detail

using detail::message;

// The enumerator's own spelling, for a caller printing a code rather than a
// sentence.  message() is what everything here prints.
[[nodiscard]] inline const char *name(const errc c) noexcept {
  return boost::describe::enum_to_string(c, "?");
}

// One implementation for both, and the same text std::format prints below.
// Written out rather than abbreviated: `decltype(e)` of a by-value `const auto`
// parameter is `const errc`, which never matches `errc`.
template <typename E>
  requires std::same_as<E, errc> || std::same_as<E, error>
std::ostream &operator<<(std::ostream &out, const E e) {
  return out << detail::message(e);
}

template <typename T> using result = std::expected<T, error>;

[[nodiscard]] constexpr std::unexpected<error> fail(const errc c) noexcept {
  return std::unexpected{error{.code = c}};
}

// The same `std::unexpected` return for results whose value type is a caller's
// tensor.  GCC's late -Wmaybe-uninitialized pass, at -O3, reads the value arm
// of the returned expected -- the union member an error return never enters --
// and reports its bytes as read, blaming the line it is built on.  Confining
// the pragma here keeps the warning live in the rest of the library.
template <typename T>
[[nodiscard]] constexpr result<T> propagate(const error e) noexcept {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  return result<T>{std::unexpected{e}};
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

template <typename T>
[[nodiscard]] constexpr bool failed_with(const result<T> &r,
                                         const errc c) noexcept {
  return !r.has_value() && r.error().code == c;
}

} // namespace einsum

// One static_assert per error code, from the same table.  `predicate` is a
// function-like macro taking an enumerator name and yielding a constant
// expression that is true when that is what went wrong.
#define EINSUM_ERRC_ASSERT_ONE(r, predicate, elem)                             \
  static_assert(!predicate(BOOST_PP_TUPLE_ELEM(0, elem)),                      \
                "einsum<\"...\">: " BOOST_PP_TUPLE_ELEM(1, elem));
#define EINSUM_ASSERT_NO_ERROR(predicate)                                      \
  BOOST_PP_SEQ_FOR_EACH(EINSUM_ERRC_ASSERT_ONE, predicate, EINSUM_ERRC_SEQ)

// From the string_view formatter, so a caller's "{:>16}" reaches the text.
template <>
struct std::formatter<einsum::errc, char>
    : std::formatter<std::string_view, char> {
  auto format(const einsum::errc c, std::format_context &ctx) const {
    return std::formatter<std::string_view, char>::format(
        einsum::detail::message(c), ctx);
  }
};

template <>
struct std::formatter<einsum::error, char>
    : std::formatter<einsum::errc, char> {
  auto format(const einsum::error e, std::format_context &ctx) const {
    return std::formatter<einsum::errc, char>::format(e.code, ctx);
  }
};
