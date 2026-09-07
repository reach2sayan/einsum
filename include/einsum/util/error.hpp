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

// The one table.  It generates the enumerators, the message strings, and -- in
// ct/subscripts.hpp -- one static_assert per code, so a subscript rejected at
// compile time reports the same sentence a runtime parse would have returned.
// A Boost.PP sequence of (name, message) tuples rather than an X-macro, because
// the compile-time side has to iterate it inside a class body.
//
// clang-format off
#define EINSUM_ERRC_SEQ                                                                          \
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
  ((output_mismatch,        "the output's rank or extents are not the ones the subscript implies"))
// clang-format on

// The enumerator names on their own, so both the enum and the Describe
// annotation below are generated from the one table rather than restating it.
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

[[nodiscard]] inline const char *name(const errc c) noexcept {
  return boost::describe::enum_to_string(c, "?");
}

// One implementation for both, and the same text std::format below prints: a
// code and the error carrying it must never read differently.  Written out
// rather than abbreviated, because `decltype(e)` of a by-value `const auto`
// parameter is `const errc`, which is not `errc` and never matches.
template <typename E>
  requires std::same_as<E, errc> || std::same_as<E, error>
std::ostream &operator<<(std::ostream &out, const E e) {
  return out << detail::message(e);
}

template <typename T> using result = std::expected<T, error>;

[[nodiscard]] constexpr std::unexpected<error> fail(const errc c) noexcept {
  return std::unexpected{error{.code = c}};
}

// The same `std::unexpected` return, written once, for the results whose value
// type is a caller's tensor.  GCC's late -Wmaybe-uninitialized pass, at -O3,
// looks at the value arm of the returned expected -- the union member an error
// return never enters -- and reports the bytes it would have held as read; the
// diagnostic is attributed to the line the expected is built on, so the pragma
// has to sit on that line.  Confining it here keeps the warning live in the
// rest of the library.
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

// One static_assert per error code, generated from the same table the codes
// are: a compile-time einsum reports the sentence a runtime one would have
// returned, and a code added to EINSUM_ERRC_SEQ is diagnosed here for free.
//
// `predicate` is a function-like macro taking an enumerator name and yielding a
// constant expression that is true when that is what went wrong.
#define EINSUM_ERRC_ASSERT_ONE(r, predicate, elem)                             \
  static_assert(!predicate(BOOST_PP_TUPLE_ELEM(0, elem)),                      \
                "einsum<\"...\">: " BOOST_PP_TUPLE_ELEM(1, elem));
#define EINSUM_ASSERT_NO_ERROR(predicate)                                      \
  BOOST_PP_SEQ_FOR_EACH(EINSUM_ERRC_ASSERT_ONE, predicate, EINSUM_ERRC_SEQ)

// Deriving from the string_view formatter, so a caller's "{:>16}" reaches the
// text.
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
