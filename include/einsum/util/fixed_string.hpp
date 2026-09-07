#pragma once

#include <algorithm>
#include <compare>
#include <cstddef>
#include <string_view>

namespace einsum::impl {

// A string usable as a non-type template parameter, which is what turns
// einsum<"ij,jk->ik">(a, b) into a type the compiler can parse the subscript
// out of.  Structural, so `data` must be public and there can be nothing else.
template <std::size_t N> struct FixedString {
  static_assert(N > 0, "FixedString: N counts the terminating NUL, so N >= 1");

  // Public by necessity -- see the structural-type note above.
  char data[N];

  consteval FixedString(const char (&str)[N]) noexcept {
    std::copy_n(str, N, data);
  }

  // Length in characters, excluding the terminating NUL.
  [[nodiscard]] static constexpr std::size_t size() noexcept { return N - 1; }

  // constexpr, not consteval: the runtime parser reads this back to cross-check.
  [[nodiscard]] constexpr std::string_view view() const noexcept {
    return {data, size()};
  }

  // Both through view(), so == and <=> cannot disagree across lengths.
  template <std::size_t M>
  [[nodiscard]] constexpr bool
  operator==(const FixedString<M> &other) const noexcept {
    return view() == other.view();
  }

  template <std::size_t M>
  [[nodiscard]] constexpr std::strong_ordering
  operator<=>(const FixedString<M> &other) const noexcept {
    return view() <=> other.view();
  }
};

template <std::size_t N> FixedString(const char (&)[N]) -> FixedString<N>;

namespace detail {
template <typename T> inline constexpr bool is_fixed_string_v = false;
template <std::size_t N>
inline constexpr bool is_fixed_string_v<FixedString<N>> = true;
} // namespace detail

template <typename T>
concept CFixedString = detail::is_fixed_string_v<T>;

namespace detail {

template <typename T>
concept CIsExactlyItsChars = (sizeof(T) == T::size() + 1);

static_assert(CIsExactlyItsChars<FixedString<1>>,
              "FixedString has grown beyond its character array: check that "
              "nothing was added to the class and that it gained no base");
static_assert(CIsExactlyItsChars<FixedString<2>>, "see above");
static_assert(CIsExactlyItsChars<FixedString<8>>, "see above");

// Two spellings of the same subscript must name the same specialisation, and
// two different ones must not.  That is the whole contract einsum<S> rests on.
template <FixedString S> struct nttp_probe {
  static constexpr auto label = S;
};

static_assert(nttp_probe<"ij">::label == FixedString{"ij"});
static_assert(std::same_as<nttp_probe<"ij">, nttp_probe<FixedString{"ij"}>>);
static_assert(!std::same_as<nttp_probe<"ij">, nttp_probe<"ik">>);
static_assert(!std::same_as<nttp_probe<"i">, nttp_probe<"ij">>);

} // namespace detail

} // namespace einsum::impl
