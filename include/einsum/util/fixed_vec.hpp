#pragma once

#include <boost/stl_interfaces/sequence_container_interface.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <ranges>

namespace einsum::impl {

// A vector with its capacity in the type and no allocator, kept only where
// constexpr forces it: a Shape has to be structural to be a template argument,
// and a Plan has to be built in a constant expression.  Every runtime-only
// sequence here is a boost::container::static_vector instead.
//
// A strict subset of std::inplace_vector<T, N>, name for name, so the day a
// toolchain here ships <inplace_vector> this file becomes a using-declaration.
// Hence push_back's precondition rather than a bool return, and try_push_back
// answering a pointer for the callers that turn a null into an errc.
//
// begin() and end() are the only members that know the storage; everything else
// a contiguous container owes its callers is what Boost.STLInterfaces derives
// from those two.  data_ and size_ are public because a structural type may
// have no private member.
template <std::copyable T, std::size_t N>
struct FixedVec
    : boost::stl_interfaces::sequence_container_interface<
          FixedVec<T, N>, boost::stl_interfaces::element_layout::contiguous> {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = typename std::array<T, N>::iterator;
  using const_iterator = typename std::array<T, N>::const_iterator;
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
  // must not compete.  A range longer than N stops at N, and the Plan that
  // meets it answers rank_mismatch.
  template <std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, T>
  constexpr FixedVec(std::from_range_t, R &&values) noexcept {
    const auto [_, last] =
        std::ranges::copy(values | std::views::take(N), data_.begin());
    size_ = static_cast<size_type>(last - data_.begin());
  }

  [[nodiscard]] static constexpr size_type max_size() noexcept { return N; }
  [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }

  [[nodiscard]] constexpr iterator begin() noexcept { return data_.begin(); }
  [[nodiscard]] constexpr const_iterator begin() const noexcept {
    return data_.begin();
  }
  [[nodiscard]] constexpr iterator end() noexcept {
    return begin() + static_cast<difference_type>(size_);
  }
  [[nodiscard]] constexpr const_iterator end() const noexcept {
    return begin() + static_cast<difference_type>(size_);
  }

  // Precondition: size() < capacity().  The caller that cannot promise that
  // asks try_push_back instead.  Stated to the optimiser as well, or GCC's
  // -Wstringop-overflow reads the unconstrained size_ and reports the last
  // element of a full array as a write past its end.
  constexpr reference push_back(const T &value) noexcept {
    [[assume(size_ < N)]];
    *end() = value;
    ++size_;
    return this->back();
  }

  constexpr pointer try_push_back(const T &value) noexcept {
    return size_ == N ? nullptr : std::addressof(push_back(value));
  }

  constexpr void pop_back() noexcept { --size_; }
  constexpr void clear() noexcept { size_ = 0; }
};

// The interface is only worth having if the standard algorithms accept it.
// Comparison is over the live prefix only -- two vectors agreeing on their
// elements are equal whatever their arrays' tails hold -- which is what the
// inherited operator== does and a defaulted one would not.
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
         FixedVec<int, 4>{1, 2} < v && v.front() == 1 && v.back() == 3 &&
         v.size() == 3 && v.data()[0] == 1 && !v.empty();
}());

} // namespace einsum::impl
