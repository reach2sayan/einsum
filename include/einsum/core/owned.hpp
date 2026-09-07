#pragma once

#include "einsum/core/view.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"

#include <Eigen/Core>
#include <experimental/mdspan>

#include <algorithm>
#include <array>
#include <ranges>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsum {

// The scratch a Plan needs, owned or borrowed.  One allocation at bind time and
// none afterwards is the whole contract; borrow() is for callers who will not
// even have that one.
template <CScalar T> class Workspace {
public:
  constexpr Workspace() noexcept = default;

  [[nodiscard]] static result<Workspace> make(const std::size_t elems) {
    Workspace ws;
    ws.owned_.resize(elems);
    return ws;
  }

  [[nodiscard]] static result<Workspace> borrow(const std::span<T> memory,
                                                const std::size_t elems) noexcept {
    if (memory.size() < elems) {
      return fail(errc::workspace_too_small);
    }
    Workspace ws;
    ws.borrowed_ = memory.first(elems);
    ws.is_borrowed_ = true;
    return ws;
  }

  // Answered from the vector each time rather than cached, so moving a
  // Workspace cannot leave a span pointing at the buffer it used to own.
  [[nodiscard]] std::span<T> data() noexcept {
    return is_borrowed_ ? borrowed_ : std::span<T>{owned_};
  }

private:
  std::vector<T> owned_{};
  std::span<T> borrowed_{};
  bool is_borrowed_ = false;
};

// The one owned result.  Storage is std::array<T, N> on the compile-time path
// and std::vector<T> on the runtime one, and that is the only difference
// between them: same layout, same view, same accessors, one evaluate body
// writing into the TensorView that view() answers.
template <CScalar T, typename Storage> class Owned {
public:
  using value_type = T;

  constexpr Owned() = default;
  constexpr explicit Owned(const Layout &layout) noexcept : layout_{layout} {}

  [[nodiscard]] static result<Owned> make(const Shape &shape)
    requires requires(Storage s) { s.resize(std::size_t{}); }
  {
    Owned out{make_layout<RowMajor>(shape)};
    out.data_.resize(static_cast<std::size_t>(out.layout_.size()));
    return out;
  }

  [[nodiscard]] constexpr TensorView<T> view() noexcept { return {data_.data(), layout_}; }
  [[nodiscard]] constexpr const Layout &layout() const noexcept { return layout_; }
  [[nodiscard]] constexpr const Shape &shape() const noexcept { return layout_.shape; }
  [[nodiscard]] constexpr const Storage &storage() const noexcept { return data_; }
  [[nodiscard]] constexpr Storage take_storage() && noexcept { return std::move(data_); }

  // The three ways to read it back, one per rung of the operand ladder.
  [[nodiscard]] constexpr std::span<const T> as_span() const noexcept { return data_; }

  // Row-major, and the subscript's rank rather than the matrix's: a rank-1
  // result is a column and a rank-0 one is 1x1.
  [[nodiscard]] result<Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                                      Eigen::RowMajor>>>
  as_matrix() const noexcept {
    if (layout_.rank() > 2) {
      return fail(errc::not_matrix);
    }
    const auto rows = static_cast<Eigen::Index>(layout_.rank() >= 1 ? layout_.shape[0] : 1);
    const auto cols = static_cast<Eigen::Index>(layout_.rank() == 2 ? layout_.shape[1] : 1);
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>{
        data_.data(), rows, cols};
  }

  // An mdspan's rank is in its type and a runtime subscript's is not, so the
  // caller names the one they expect.
  template <std::size_t R>
  [[nodiscard]] result<std::mdspan<const T, std::dextents<std::size_t, R>>>
  as_mdspan() const noexcept {
    if (layout_.rank() != R) {
      return fail(errc::rank_mismatch);
    }
    std::array<std::size_t, R> extents{};
    std::ranges::transform(layout_.shape, extents.begin(),
                           [](const index_t e) { return static_cast<std::size_t>(e); });
    return std::mdspan<const T, std::dextents<std::size_t, R>>{data_.data(), extents};
  }

private:
  Storage data_{};
  Layout layout_{};
};

template <CScalar T> using OwnedBuffer = Owned<T, std::vector<T>>;
template <CScalar T, std::size_t N> using OwnedArray = Owned<T, std::array<T, N>>;

// Which accessor "the same kind as the inputs" means.  Each one hands back
// something that owns its elements: a Map or a span into a buffer the call is
// about to drop would dangle, so only the mdspan rung -- whose rank is not in
// any type a runtime subscript could name -- returns the buffer itself.
template <OperandKind K> struct result_for;

// The one copy in the library, and it cannot be avoided: an Eigen::Matrix owns
// its elements through its own allocator and has no way to adopt a buffer, so
// handing back a Matrix means copying into one.  The alternative -- returning a
// Map into the buffer -- would dangle the moment the call returned.
template <> struct result_for<OperandKind::eigen> {
  [[nodiscard]] static auto get(auto &&owned) {
    using T = typename std::remove_cvref_t<decltype(owned)>::value_type;
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    return owned.as_matrix().transform([](const auto &m) { return Matrix{m}; });
  }
};

template <> struct result_for<OperandKind::mdspan> {
  [[nodiscard]] static auto get(auto &&owned) {
    return result<std::remove_cvref_t<decltype(owned)>>{std::forward<decltype(owned)>(owned)};
  }
};

// Moved, not copied: the buffer already is the std::vector the caller asked for.
template <> struct result_for<OperandKind::flat> {
  [[nodiscard]] static auto get(auto &&owned) {
    using T = typename std::remove_cvref_t<decltype(owned)>::value_type;
    return result<std::vector<T>>{std::forward<decltype(owned)>(owned).take_storage()};
  }
};

} // namespace einsum
