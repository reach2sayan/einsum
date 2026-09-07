#pragma once

#include "einsum/core/execute.hpp"
#include "einsum/core/kind.hpp"
#include "einsum/core/lower.hpp"
#include "einsum/core/owned.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace einsum {

namespace impl {

template <typename X> inline constexpr bool is_span_v = false;
template <typename U, std::size_t N> inline constexpr bool is_span_v<std::span<U, N>> = true;

// Every operand shape, funnelled to the one the executor wants.  The fallible
// forms carry their error from here rather than from the call, which is what
// lets a caller pass flat(v, {2, 3}) straight through.
template <CScalar T, typename X>
[[nodiscard]] constexpr result<TensorView<const T>> to_input(X &&x) noexcept {
  static_assert(std::same_as<scalar_of_t<X>, T>,
                "einsum: every operand and the output must have the same scalar type");
  const auto as_const = [](const auto &v) noexcept {
    return TensorView<const T>{v.data, v.layout};
  };
  using B = std::remove_cvref_t<X>;
  if constexpr (is_view_result_v<B>) {
    return x.transform(as_const);
  } else if constexpr (is_tensor_view_v<B>) {
    return as_const(x);
  } else {
    return as_const(as_view(x));
  }
}

template <typename X> [[nodiscard]] constexpr auto to_into(X &&x) noexcept {
  using B = std::remove_cvref_t<X>;
  if constexpr (is_into_v<B>) {
    return result<B>{x};
  } else {
    return std::forward<X>(x);
  }
}

template <CScalar T, typename... In>
[[nodiscard]] result<FixedVec<TensorView<const T>, kMaxOperands>>
collect_views(const Plan &plan, In &&...ins) noexcept {
  if (sizeof...(In) != plan.operand_count()) {
    return fail(errc::operand_count_mismatch);
  }
  const std::array<result<TensorView<const T>>, sizeof...(In)> converted{
      to_input<T>(EINSUM_FWD(ins))...};
  FixedVec<TensorView<const T>, kMaxOperands> views;
  for (const auto &view : converted) {
    if (!view) {
      return std::unexpected{view.error()};
    }
    (void)views.push_back(*view);
  }
  return views;
}

} // namespace impl

// A Plan that has met its operands.  Everything variable has been resolved:
// eval() is a sequence of Eigen calls over pointers this object already holds,
// and it allocates nothing.
template <CScalar T> class Bound {
public:
  using value_type = T;

  Bound() = default;
  Bound(Plan plan, impl::Geometry geometry,
        impl::FixedVec<TensorView<const T>, kMaxOperands> inputs, TensorView<T> out,
        Workspace<T> workspace) noexcept
      : plan_{std::move(plan)}, geom_{geometry}, ins_{inputs}, out_{out},
        ws_{std::move(workspace)} {}

  [[nodiscard]] const Plan &plan() const noexcept { return plan_; }
  [[nodiscard]] const Shape &output_shape() const noexcept { return geom_.out_shape; }
  [[nodiscard]] std::size_t scratch_bytes() const noexcept {
    return static_cast<std::size_t>(geom_.scratch_elems) * sizeof(T);
  }
  [[nodiscard]] std::span<T> workspace_data() noexcept { return ws_.data(); }
  [[nodiscard]] const TensorView<T> &output() const noexcept { return out_; }

  // Nothing here can fail: every question that could have been answered "no"
  // was answered at bind time.
  void eval() noexcept { impl::execute<T>(plan_, geom_, ins_.span(), out_, ws_.data()); }

  // The same computation over different memory of the same shape.  Strides too,
  // not just extents: a Geometry is a set of offsets, and an operand that moved
  // its rows is a different one however alike its shape looks.
  template <typename... A> [[nodiscard]] result<void> eval(A &&...args) noexcept;

private:
  Plan plan_{};
  impl::Geometry geom_{};
  impl::FixedVec<TensorView<const T>, kMaxOperands> ins_{};
  TensorView<T> out_{};
  Workspace<T> ws_{};
};

namespace impl {

template <CScalar T>
[[nodiscard]] constexpr FixedVec<Layout, kMaxOperands>
layouts_of(const FixedVec<TensorView<const T>, kMaxOperands> &views) noexcept {
  FixedVec<Layout, kMaxOperands> layouts;
  for (const auto &view : views) {
    (void)layouts.push_back(view.layout);
  }
  return layouts;
}

// The one place a Bound is built.  `borrow` says which Workspace factory to
// use; `ws` is empty and unread when it is false.
template <CScalar T>
[[nodiscard]] result<Bound<T>>
bind_prepared(const Plan &plan, const FixedVec<TensorView<const T>, kMaxOperands> &views,
              const TensorView<T> &out, const std::span<T> ws, const bool borrow) {
  return make_geometry(plan, layouts_of<T>(views).span(), out.layout)
      .and_then([&](const Geometry &geom) -> result<Bound<T>> {
        const auto elems = static_cast<std::size_t>(geom.scratch_elems);
        return (borrow ? Workspace<T>::borrow(ws, elems) : Workspace<T>::make(elems))
            .transform([&](Workspace<T> &&workspace) {
              return Bound<T>{plan, geom, views, out, std::move(workspace)};
            });
      });
}

// bind(ops..., into(out) [, workspace]): the output tag marks where the
// operands stop, and a trailing std::span is the optional workspace.
template <std::size_t IntoAt, typename Tup, std::size_t... I>
[[nodiscard]] auto bind_from_tuple(const Plan &plan, Tup &tup, std::index_sequence<I...>,
                                   const bool borrow) {
  auto out = to_into(std::get<IntoAt>(tup));
  using T = typename std::remove_cvref_t<decltype(*out)>::value_type;
  std::span<T> ws{};
  if constexpr (IntoAt + 1 < std::tuple_size_v<std::remove_cvref_t<Tup>>) {
    ws = std::span<T>{std::get<IntoAt + 1>(tup)};
  }
  if (!out) {
    return result<Bound<T>>{std::unexpected{out.error()}};
  }
  return collect_views<T>(plan, std::get<I>(tup)...)
      .and_then([&](const FixedVec<TensorView<const T>, kMaxOperands> &views) {
        return bind_prepared<T>(plan, views, out->view, ws, borrow);
      });
}

// eval(ops...) with no output tag: the operands' own kind decides what comes
// back, and this makes a buffer of it.  The evaluate path is the same one --
// only where out_ points differs.  K is a parameter because to_matrix() wants
// the Eigen accessor whatever the operands were.
template <OperandKind K, typename Tup, std::size_t... I>
[[nodiscard]] auto eval_owning(const Plan &plan, Tup &tup, std::index_sequence<I...>) {
  using T = common_value_t<std::remove_cvref_t<std::tuple_element_t<I, Tup>>...>;
  using Views = FixedVec<TensorView<const T>, kMaxOperands>;

  // Each step's failure is the whole call's, so the chain says so once rather
  // than four times.
  return collect_views<T>(plan, std::get<I>(tup)...)
      .and_then([&](const Views &views) {
        return infer_output_shape(plan, layouts_of<T>(views).span())
            .and_then(OwnedBuffer<T>::make)
            .and_then([&](OwnedBuffer<T> &&owned) {
              return bind_prepared<T>(plan, views, owned.view(), std::span<T>{}, false)
                  .and_then([&](Bound<T> &&bound) {
                    bound.eval();
                    return result_for<K>::get(std::move(owned));
                  });
            });
      });
}

// Where the operands stop.  A trailing std::span is the workspace; what sits
// before it is either an into() tag or the last operand, and that is the whole
// of the difference between the two call shapes.
template <typename... A> struct call_shape {
  static constexpr std::size_t count = sizeof...(A);
  static_assert(count >= 1, "einsum: takes the operands, then optionally into(output), "
                            "then optionally a std::span workspace");
  static constexpr bool workspace =
      is_span_v<std::remove_cvref_t<std::tuple_element_t<count - 1, std::tuple<A...>>>>;
  static_assert(!workspace || count >= 2, "einsum: a workspace comes after the operands");
  static constexpr std::size_t tail = workspace ? count - 2 : count - 1;
  static constexpr bool explicit_output = CInto<std::tuple_element_t<tail, std::tuple<A...>>>;
  static constexpr std::size_t operands = explicit_output ? tail : tail + 1;

  static_assert(explicit_output || !workspace,
                "einsum: a borrowed workspace goes with into(...); the form that makes "
                "the result for you allocates it either way");
  static_assert(explicit_output ||
                    boost::mp11::mp_apply<
                        homogeneous,
                        boost::mp11::mp_take_c<boost::mp11::mp_list<std::remove_cvref_t<A>...>,
                                               operands>>::value,
                "einsum: every operand of one call must be the same kind -- all Eigen "
                "objects, all mdspans, or all flat(range or pointer, {shape}) -- and all "
                "the same scalar type");
};

template <typename... A> [[nodiscard]] auto bind_dispatch(const Plan &plan, A &&...args) {
  using shape = call_shape<A...>;
  static_assert(shape::explicit_output,
                "Plan::bind: name the output with into(...); the form that makes one for "
                "you is Plan::eval / operator(), which cannot hand back a reusable object "
                "and the buffer it writes into at the same time");
  auto tup = std::forward_as_tuple(std::forward<A>(args)...);
  // if constexpr, not just the assert: a failed static_assert does not stop the
  // body being instantiated, and this one would then fail again in its own words.
  if constexpr (shape::explicit_output) {
    return bind_from_tuple<shape::tail>(plan, tup, std::make_index_sequence<shape::tail>{},
                                        shape::workspace);
  } else {
    return result<void>{};
  }
}

template <typename... A> [[nodiscard]] auto eval_dispatch(const Plan &plan, A &&...args) {
  using shape = call_shape<A...>;
  auto tup = std::forward_as_tuple(std::forward<A>(args)...);
  if constexpr (shape::explicit_output) {
    return bind_from_tuple<shape::tail>(plan, tup, std::make_index_sequence<shape::tail>{},
                                        shape::workspace)
        .transform([](auto &&bound) noexcept { bound.eval(); });
  } else {
    // Without an output tag every argument is an operand, so the kind is theirs.
    return eval_owning<common_kind_v<std::remove_cvref_t<A>...>>(
        plan, tup, std::make_index_sequence<shape::operands>{});
  }
}

// The rebinding counterpart: same shapes, new pointers.
template <CScalar T, std::size_t IntoAt, typename Tup, std::size_t... I>
[[nodiscard]] result<void>
rebind_from_tuple(const Plan &plan, FixedVec<TensorView<const T>, kMaxOperands> &ins,
                  TensorView<T> &out, Tup &tup, std::index_sequence<I...>) noexcept {
  const auto target = to_into(std::get<IntoAt>(tup));
  if (!target) {
    return std::unexpected{target.error()};
  }
  const auto views = collect_views<T>(plan, std::get<I>(tup)...);
  if (!views) {
    return std::unexpected{views.error()};
  }
  const auto paired = std::views::zip(*views, ins);
  if (std::ranges::any_of(paired, [](const auto &pair) {
        const auto &[fresh, bound] = pair;
        return fresh.layout.rank() != bound.layout.rank();
      })) {
    return fail(errc::rank_mismatch);
  }
  // Strides too, not just extents: a Geometry is a set of offsets, and an
  // operand that moved its rows is a different one however alike its shape is.
  if (std::ranges::any_of(paired, [](const auto &pair) {
        const auto &[fresh, bound] = pair;
        return fresh.layout != bound.layout;
      })) {
    return fail(errc::extent_conflict);
  }
  if (target->view.layout != out.layout) {
    return fail(errc::output_mismatch);
  }
  std::ranges::for_each(paired, [](auto pair) {
    auto &[fresh, bound] = pair;
    bound.data = fresh.data;
  });
  out.data = target->view.data;
  return {};
}

} // namespace impl

template <CScalar T>
template <typename... A>
result<void> Bound<T>::eval(A &&...args) noexcept {
  constexpr std::size_t n = sizeof...(A);
  static_assert(n >= 2, "Bound::eval: takes the operands, then into(output)");
  auto tup = std::forward_as_tuple(std::forward<A>(args)...);
  return impl::rebind_from_tuple<T, n - 1>(plan_, ins_, out_, tup,
                                           std::make_index_sequence<n - 1>{})
      .transform([this]() noexcept { eval(); });
}

// --- the Plan call surface ---------------------------------------------------
// bind() keeps the object for repeated evaluation and needs into(out).
// operator() and eval() are the one-shot form: with into(out) they answer
// result<void>, without it the result in the operands' own kind.  to_matrix()
// is the same call reading the buffer back as an Eigen matrix whatever the
// operands were.
template <typename... A> auto Plan::bind(A &&...args) const {
  return impl::bind_dispatch(*this, std::forward<A>(args)...);
}

template <typename... A> auto Plan::eval(A &&...args) const {
  return impl::eval_dispatch(*this, std::forward<A>(args)...);
}

template <typename... A> auto Plan::operator()(A &&...args) const {
  return eval(std::forward<A>(args)...);
}

template <typename... A> auto Plan::to_matrix(A &&...inputs) const {
  auto tup = std::forward_as_tuple(std::forward<A>(inputs)...);
  return impl::eval_owning<OperandKind::eigen>(*this, tup,
                                               std::make_index_sequence<sizeof...(A)>{});
}

} // namespace einsum
