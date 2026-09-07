#pragma once

#include "einsum/core/kind.hpp"
#include "einsum/core/limits.hpp"
#include "einsum/core/view.hpp"

#include <boost/mp11/list.hpp>

#include <Eigen/Core>
#include <experimental/mdspan>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

// The compile-time path's operands and the few places it needs a type rather
// than a value.  "Which scalar" and "which kind" are core/kind.hpp's questions
// and are not asked again here.
namespace einsum::ct {

// Only the operands whose whole Layout is known without an object: that is what
// lets einsum<"ij,jk->ik">(a, b) size its result array and its scratch.
template <typename Op> struct operand_traits;

template <typename T, typename E, typename L, typename A>
  requires std::same_as<A, std::default_accessor<T>> && (E::rank_dynamic() == 0) &&
           (std::same_as<L, std::layout_right> || std::same_as<L, std::layout_left>)
struct operand_traits<std::mdspan<T, E, L, A>> {
  using value_type = std::remove_const_t<T>;
  using layout_policy = std::conditional_t<std::same_as<L, std::layout_right>, RowMajor, ColMajor>;
  static constexpr Layout layout = [] {
    Shape shape;
    for (std::size_t i = 0; i < E::rank(); ++i) {
      (void)shape.push_back(static_cast<index_t>(E::static_extent(i)));
    }
    return make_layout<layout_policy>(shape);
  }();
  static_assert(E::rank() <= kMaxRank,
                "einsum<\"...\">: the mdspan has more extents than einsum::kMaxRank");
};

// A fixed-size vector is rank 1 and its order says nothing; a matrix takes
// Eigen's, which is column-major unless the option says otherwise.
template <typename T, int R, int C, int Opts, int MR, int MC>
  requires(R != Eigen::Dynamic && C != Eigen::Dynamic)
struct operand_traits<Eigen::Matrix<T, R, C, Opts, MR, MC>> {
  using value_type = std::remove_const_t<T>;
  using layout_policy =
      std::conditional_t<(R == 1 || C == 1) || (Opts & Eigen::RowMajor) != 0, RowMajor, ColMajor>;
  static constexpr Layout layout = [] {
    Shape shape;
    if constexpr (R == 1 || C == 1) {
      (void)shape.push_back(index_t{R} * index_t{C});
    } else {
      (void)shape.push_back(index_t{R});
      (void)shape.push_back(index_t{C});
    }
    return make_layout<layout_policy>(shape);
  }();
};

template <typename Op>
concept CStaticOperand = requires {
  typename operand_traits<std::remove_cvref_t<Op>>::value_type;
  { operand_traits<std::remove_cvref_t<Op>>::layout } -> std::convertible_to<Layout>;
} && CScalar<typename operand_traits<std::remove_cvref_t<Op>>::value_type>;

// The order the first operand is in, which is the order an Eigen-kind result
// comes back in.  The operands are all one kind by then, but not necessarily
// all one order, so one of them has to be the one that says.
template <typename... Ops>
using first_layout_policy_t = typename operand_traits<
    std::remove_cvref_t<boost::mp11::mp_front<boost::mp11::mp_list<Ops...>>>>::layout_policy;

namespace detail {
// A Shape is structural, so the result's extents can be read straight off the
// one the lowering computed -- no second description of the same shape.
template <Shape S, std::size_t... I>
[[nodiscard]] constexpr auto extents_from(std::index_sequence<I...>) noexcept
    -> std::extents<std::size_t, static_cast<std::size_t>(S.data_[I])...> {
  return {};
}
} // namespace detail

template <Shape S>
using extents_of_t = decltype(detail::extents_from<S>(std::make_index_sequence<S.size_>{}));

static_assert(std::same_as<extents_of_t<Shape{2, 3}>, std::extents<std::size_t, 2, 3>>);
static_assert(std::same_as<extents_of_t<Shape{}>, std::extents<std::size_t>>);

} // namespace einsum::ct
