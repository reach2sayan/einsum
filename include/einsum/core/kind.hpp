#pragma once

#include "einsum/core/view.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>

#include <cstdint>
#include <type_traits>

// What kind of thing an operand is, decided by the concepts it satisfies rather
// than by a tag anyone has to remember to write.  Three rungs, because there are
// three ways memory can describe itself: Eigen's, mdspan's, and flat.  The kind
// decides that every operand of one call matches its neighbours, and what kind
// the result comes back as.
namespace einsum {

enum class OperandKind : std::uint8_t { eigen, mdspan, flat };

template <typename X>
concept CEigenOperand = impl::CEigenDense<std::remove_cvref_t<X>>;

// The three member types the standard gives an mdspan; naming them rather than
// the class template lets a conforming reimplementation through.
template <typename X>
concept CMdspanOperand = !CEigenOperand<X> && requires {
  typename std::remove_cvref_t<X>::extents_type;
  typename std::remove_cvref_t<X>::mapping_type;
  typename std::remove_cvref_t<X>::accessor_type;
};

// Anything as_view() reaches, which after flat() is everything else: a
// TensorView, or one an adaptor could not build.
template <typename X>
concept COperand = requires { typename impl::view_of_t<X>; };

namespace impl {
template <OperandKind K> using kind_constant = std::integral_constant<OperandKind, K>;
} // namespace impl

template <COperand X>
inline constexpr OperandKind kind_of_v = boost::mp11::mp_cond<
    boost::mp11::mp_bool<CEigenOperand<X>>, impl::kind_constant<OperandKind::eigen>,
    boost::mp11::mp_bool<CMdspanOperand<X>>, impl::kind_constant<OperandKind::mdspan>,
    boost::mp11::mp_true, impl::kind_constant<OperandKind::flat>>::value;

// One kind and one scalar across the whole call.  Nothing about a mixed call
// depends on the data, so nothing about it should wait for run time.
template <typename... Ops>
concept CHomogeneous =
    sizeof...(Ops) > 0 && (COperand<Ops> && ...) &&
    boost::mp11::mp_apply<boost::mp11::mp_same,
                          boost::mp11::mp_list<impl::scalar_of_t<Ops>...>>::value &&
    boost::mp11::mp_apply<
        boost::mp11::mp_same,
        boost::mp11::mp_list<impl::kind_constant<kind_of_v<Ops>>...>>::value;

// The concept as a trait, so a pack held in an mp_list can be asked.
namespace impl {
template <typename... Ops> struct homogeneous : std::bool_constant<CHomogeneous<Ops...>> {};
} // namespace impl

template <typename... Ops>
  requires CHomogeneous<Ops...>
using common_value_t = boost::mp11::mp_front<boost::mp11::mp_list<impl::scalar_of_t<Ops>...>>;

template <typename... Ops>
  requires CHomogeneous<Ops...>
inline constexpr OperandKind
    common_kind_v = kind_of_v<boost::mp11::mp_front<boost::mp11::mp_list<Ops...>>>;

} // namespace einsum
