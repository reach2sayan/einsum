#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <ranges>
#include <type_traits>

namespace einsum::impl {

// A vector with its capacity in the type and no allocator, kept only where
// constexpr forces it: a Shape has to be structural so it can be a template
// argument (ct/labels.hpp turns one into std::extents), and a Plan has to be
// built in a constant expression.  Every runtime-only sequence here is a
// boost::container::static_vector instead.
//
// The interface is a strict subset of std::inplace_vector<T, N>, name for name
// and meaning for meaning, because that is what this is -- no toolchain here
// ships <inplace_vector> yet, and the day one does this file becomes a single
// using-declaration.  Which is why push_back has inplace_vector's precondition
// rather than a bool return, and why the fallible form is try_push_back
// answering a pointer: an einsum that cannot grow a Shape turns a null into an
// errc, and that is the only place the difference is visible.
//
// data_ and size_ are public because a structural type has no invariant to
// protect, and no member may be private.
template <typename T, std::size_t N> struct FixedVec {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  std::array<T, N> data_{};
  size_type size_ = 0;

  constexpr FixedVec() noexcept = default;
  constexpr explicit FixedVec(const size_type n) noexcept
      : size_{n < N ? n : N} {}
  constexpr FixedVec(const std::initializer_list<T> values) noexcept
      : FixedVec(std::from_range, values) {}

  // std::from_range, not a bare range constructor: `Shape{r}` and `Shape{2, 3}`
  // must not compete, and inplace_vector spells the range one this way.  A
  // range longer than N stops at N, and the Plan that meets it answers
  // rank_mismatch.
  template <std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, T>
  constexpr FixedVec(std::from_range_t, R &&values) noexcept {
    for (auto &&value : values) {
      if (size_ == N) {
        return;
      }
      push_back(static_cast<T>(value));
    }
  }

  [[nodiscard]] static constexpr size_type max_size() noexcept { return N; }
  [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }
  [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] constexpr pointer data() noexcept { return data_.data(); }
  [[nodiscard]] constexpr const_pointer data() const noexcept {
    return data_.data();
  }
  [[nodiscard]] constexpr iterator begin() noexcept { return data(); }
  [[nodiscard]] constexpr iterator end() noexcept { return data() + size_; }
  [[nodiscard]] constexpr const_iterator begin() const noexcept {
    return data();
  }
  [[nodiscard]] constexpr const_iterator end() const noexcept {
    return data() + size_;
  }
  [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
    return begin();
  }
  [[nodiscard]] constexpr const_iterator cend() const noexcept { return end(); }
  [[nodiscard]] constexpr reverse_iterator rbegin() noexcept {
    return reverse_iterator{end()};
  }
  [[nodiscard]] constexpr reverse_iterator rend() noexcept {
    return reverse_iterator{begin()};
  }
  [[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator{end()};
  }
  [[nodiscard]] constexpr const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator{begin()};
  }

  [[nodiscard]] constexpr reference operator[](const size_type i) noexcept {
    return data_[i];
  }
  [[nodiscard]] constexpr const_reference
  operator[](const size_type i) const noexcept {
    return data_[i];
  }
  [[nodiscard]] constexpr reference front() noexcept { return data_[0]; }
  [[nodiscard]] constexpr const_reference front() const noexcept {
    return data_[0];
  }
  [[nodiscard]] constexpr reference back() noexcept { return data_[size_ - 1]; }
  [[nodiscard]] constexpr const_reference back() const noexcept {
    return data_[size_ - 1];
  }

  // Precondition: size() < capacity().  The caller that cannot promise that
  // asks try_push_back instead.
  constexpr reference push_back(const T &value) noexcept {
    data_[size_++] = value;
    return back();
  }

  constexpr pointer try_push_back(const T &value) noexcept {
    return size_ == N ? nullptr : std::addressof(push_back(value));
  }

  constexpr void pop_back() noexcept { --size_; }
  constexpr void clear() noexcept { size_ = 0; }

  // Over the live prefix only: two vectors that agree on their elements are
  // equal whatever the tails of their arrays still hold, which is why neither
  // of these can be defaulted.
  [[nodiscard]] constexpr bool
  operator==(const FixedVec &other) const noexcept {
    return std::ranges::equal(*this, other);
  }
  [[nodiscard]] constexpr auto operator<=>(const FixedVec &other) const noexcept
    requires std::three_way_comparable<T>
  {
    return std::lexicographical_compare_three_way(begin(), end(), other.begin(),
                                                  other.end());
  }
};

// The interface above is only worth having if the standard algorithms accept
// it, so this is what those asserts are for.
static_assert(std::ranges::contiguous_range<FixedVec<int, 4>>);
static_assert(std::ranges::sized_range<FixedVec<int, 4>>);
static_assert(std::ranges::common_range<FixedVec<int, 4>>);
static_assert([] {
  FixedVec<int, 4> v{3, 1, 2};
  std::ranges::sort(v);
  FixedVec<int, 4> copy;
  copy.size_ = v.size();
  std::ranges::copy(v, copy.begin());
  return std::ranges::find(v, 2) != v.end() && std::ranges::contains(v, 3) &&
         std::ranges::fold_left(v, 0, std::plus<>{}) == 6 && copy == v &&
         v.front() == 1 && v.back() == 3;
}());

} // namespace einsum::impl
