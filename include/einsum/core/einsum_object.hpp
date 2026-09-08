#pragma once

#include "einsum/core/execute.hpp"
#include "einsum/core/kind.hpp"
#include "einsum/core/lower.hpp"
#include "einsum/core/path.hpp"
#include "einsum/core/owned.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <boost/container/static_vector.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsum {

namespace impl {

template <typename S>
concept CScratch = requires(const S &s, std::size_t n) {
  { s.bytes(n) } -> std::same_as<std::span<std::byte>>;
};

class HeapScratch {
public:
  [[nodiscard]] std::span<std::byte> bytes(const std::size_t n) const {
    if (buffer_.size() < n) {
      buffer_.resize(n);
    }
    return std::span{buffer_}.first(n);
  }

private:
  mutable std::vector<std::byte> buffer_;
};

template <typename R> inline constexpr char result_id = 0;

struct LastLowering {
  boost::container::static_vector<Layout, kMaxOperands> operands{};
  // The plan the last call actually ran: the subscript's own, unless it had a
  // '...', in which case it is the one expanded against those operands' ranks.
  Plan plan{};
  Shape out_shape{};
  Layout out_layout{};
  Geometry geometry{};
  const void *result_kind = nullptr;

  [[nodiscard]] bool matches(const std::span<const Layout> lays,
                             const void *kind) const noexcept {
    return result_kind == kind && std::ranges::equal(operands, lays);
  }
};

// --- one operand, ready for the executor
// -------------------------------------- An operand is either addressable where
// it already lives, or it is walked once through its accessor and packed
// row-major into scratch.  Which one it is is a property of its type; where the
// packed copy goes is decided per call.
template <CScalar T> struct Prepared {
  Layout layout{};
  const T *data = nullptr;
  index_t offset = 0; // into the scratch block, when packed
  bool packed = false;
};

using Layouts = boost::container::static_vector<Layout, kMaxOperands>;

// The layout an operand presents to the lowering: its own strides when the
// kernels can address it, row-major when it is about to be packed.
template <COperand X>
[[nodiscard]] constexpr Layout layout_for(const X &x,
                                          const Shape &shape) noexcept {
  if constexpr (CContiguous<X>) {
    if constexpr (CEigenTensor<std::remove_cvref_t<X>>) {
      return tensor_layout<std::remove_cvref_t<X>>(shape);
    } else if constexpr (CEigenDense<std::remove_cvref_t<X>>) {
      return eigen_layout(x);
    } else {
      Layout out;
      for (const auto i : std::views::iota(std::size_t{0}, rank_v<X>)) {
        out.shape.push_back(static_cast<index_t>(x.extent(i)));
        out.strides.push_back(static_cast<index_t>(x.stride(i)));
      }
      return out;
    }
  } else {
    return make_layout<RowMajor>(shape);
  }
}

// The result's own memory, described at the rank the subscript actually
// produced rather than at the rank the result type happens to have: an Eigen
// matrix holding a rank-1 result is Nx1, and the executor should see one axis,
// not two.  A fresh result is always densely packed, so the strides follow from
// the shape and the storage order alone.
template <COperand R>
[[nodiscard]] constexpr Layout layout_of_result(const Shape &shape) noexcept {
  if constexpr (CEigenTensor<R>) {
    return tensor_layout<R>(shape);
  } else if constexpr (CMdarray<R>) {
    return make_layout<RowMajor>(shape); // layout_right
  } else if constexpr (CEigenDense<R>) {
    Layout laid{.shape = shape, .strides = {}};
    if (shape.size() == 1) {
      laid.strides.push_back(1);
    } else if (shape.size() == 2) {
      laid.strides.push_back(R::IsRowMajor ? shape[1] : 1);
      laid.strides.push_back(R::IsRowMajor ? 1 : shape[0]);
    }
    return laid;
  } else {
    return make_layout<RowMajor>(shape);
  }
}

template <COperand X>
[[nodiscard]] constexpr const scalar_of_t<X> *data_of(const X &x) noexcept {
  if constexpr (CEigenDense<std::remove_cvref_t<X>>) {
    return x.data();
  } else {
    return x.data_handle();
  }
}

// Is the last argument an output rather than an operand?  Only ever asked
// where the operand count is a compile-time constant -- the compile-time
// object -- because at run time it is not, and a trailing non-const Eigen
// matrix is then indistinguishable from an operand.
// Enough of a tuple for std::tuple_size, std::tuple_element_t and std::get:
// both call operators forward their arguments as one, and nothing else is asked
// of it.
template <typename Tup>
concept CTupleLike = requires { std::tuple_size<std::remove_cvref_t<Tup>>::value; };

template <CTupleLike Tup, std::size_t... I>
[[nodiscard]] consteval bool out_form_over(std::index_sequence<I...>) noexcept {
  using Last = std::tuple_element_t<sizeof...(I), Tup>;
  if constexpr (!std::is_lvalue_reference_v<Last> ||
                std::is_const_v<std::remove_reference_t<Last>>) {
    return false;
  } else if constexpr (!CSameFamily<std::tuple_element_t<I, Tup>...>) {
    return false;
  } else {
    return std::same_as<std::remove_cvref_t<Last>,
                        result_of_t<std::tuple_element_t<I, Tup>...>>;
  }
}

template <typename... A> [[nodiscard]] consteval bool is_out_form() noexcept {
  return out_form_over<std::tuple<A...>>(
      std::make_index_sequence<sizeof...(A) - 1>{});
}

// The same question on the runtime path, where the operand count is not a
// constant and so cannot answer it.  Constness cannot answer it either: the
// ordinary `e(a, b)` over two mutable locals is exactly the shape a
// two-operands-and-an-output call has, and reading one as the other would tax
// every call site to spell what it already meant.
//
// So the output says so.  einsum::out(x) is the whole disambiguator, it is
// checked here as a type, and an untagged trailing argument is an operand
// however it was declared.
template <typename X> inline constexpr bool is_out_tag_v = false;

template <typename... A>
[[nodiscard]] consteval bool is_rt_out_form() noexcept {
  if constexpr (sizeof...(A) < 2) {
    // One argument is an operand: a subscript naming none does not parse, so
    // there is nothing an output could be the output of.
    return false;
  } else {
    return is_out_tag_v<std::remove_cvref_t<
        std::tuple_element_t<sizeof...(A) - 1, std::tuple<A...>>>>;
  }
}

// --- the one contraction -----------------------------------------------------
// What both entry points do once the lowering is known.  They differ only in
// where the block of scratch comes from -- a pooled allocation on the runtime
// path, an array in the frame on the compile-time one -- and in whether the
// offsets into it were computed or were constants; both are parameters, so the
// contraction itself is written once.

// Where each thing sits in that one block: the geometry's own working space
// first, because execute() addresses it from zero, then a packed copy of every
// operand the kernels cannot address, then the output when it is not one the
// result can be written into directly.  `total` is the block's size in
// elements.  Constexpr, so the static path sizes its frame array with it.
struct ScratchMap {
  std::array<index_t, kMaxOperands> packed_at{}; // -1 where nothing is packed
  index_t out_at = -1;                           // -1 when the output is direct
  index_t total = 0;
};

[[nodiscard]] constexpr ScratchMap
scratch_offsets(const std::span<const Layout> lays,
                const std::span<const bool> gathered, const index_t scratch_elems,
                const Shape &out_shape, const bool out_direct) noexcept {
  ScratchMap map;
  map.packed_at.fill(-1);
  index_t cursor = scratch_elems;
  for (const auto [i, packs] : gathered | std::views::enumerate) {
    if (packs) {
      map.packed_at[static_cast<std::size_t>(i)] = cursor;
      cursor += lays[static_cast<std::size_t>(i)].size();
    }
  }
  if (!out_direct) {
    map.out_at = cursor;
    cursor += product(out_shape);
  }
  map.total = cursor;
  return map;
}

// Pack what has to be packed, contract, and scatter if the result could not be
// written into directly.  `pool` is the block scratch_offsets() described.
template <CScalar T, COperand R, COperand... Ops>
void contract_into(const Plan &plan, const Geometry &geometry,
                   const std::span<const Layout> lays, const Layout &out_layout,
                   T *const pool, const ScratchMap &map, R &out,
                   const Shape &fitted, const Ops &...ops) {
  boost::container::static_vector<TensorView<const T>, kMaxOperands> views;
  std::size_t next = 0;
  const auto prepare = [&](const auto &op) {
    const T *base = nullptr;
    if constexpr (CContiguous<decltype(op)>) {
      base = data_of(op);
    } else {
      T *const packed = pool + map.packed_at[next];
      gather(op, packed, lays[next].shape);
      base = packed;
    }
    views.push_back(TensorView<const T>{base, lays[next]});
    ++next;
  };
  (prepare(ops), ...);

  // if constexpr, not a ternary: the two branches have different pointer types.
  T *target_data = nullptr;
  if constexpr (CDirectWritable<R>) {
    target_data = out.data();
  } else {
    target_data = pool + map.out_at;
  }
  execute<T>(plan, geometry, std::span<const TensorView<const T>>{views},
             TensorView<T>{target_data, out_layout},
             std::span<T>{pool, static_cast<std::size_t>(geometry.scratch_elems)});

  if constexpr (!CDirectWritable<R>) {
    scatter(static_cast<const T *>(pool + map.out_at), out, fitted);
  }
}


} // namespace impl

// --- naming an output --------------------------------------------------------
// out(x) marks the last argument of a call as where the result goes rather than
// as one more operand.  That is the only way the runtime path can be told,
// since the operand count is a value there and not a constant -- and it is what
// lets a call reach a rank no operand implies: "ij,kl->ijkl" writes rank 4 from
// two rank-2 operands, and x's own type is where that 4 is written down.
//
// A reference, deliberately: it exists for the length of one call and names
// storage the caller already has, so it binds lvalues only.
template <COperand R> struct Out {
  R &target;
};

template <COperand R> [[nodiscard]] constexpr Out<R> out(R &target) noexcept {
  return {target};
}

namespace impl {
template <typename R> inline constexpr bool is_out_tag_v<Out<R>> = true;
} // namespace impl

// --- the object
// --------------------------------------------------------------- A parsed
// subscript and nothing else a caller can see.  Both call operators are const:
// the object is immutable, so one may be shared, stored by value, or made
// constexpr on the compile-time path.  Concurrent calls on the *same* object
// still need synchronisation, because they share the scratch cache.
template <impl::CScratch Scratch> class BasicEinsum;

namespace impl {
// The one way an Einsum is built from a Plan.  A caller never has a Plan --
// einsum() hands back the finished object -- so the constructor that takes one
// is no part of the surface.
struct einsum_access;
} // namespace impl

template <impl::CScratch Scratch> class BasicEinsum {
public:
  // A copy starts cold: the caches belong to the object that warmed them, and
  // sharing them would make two objects that must not interact do so.
  BasicEinsum(const BasicEinsum &other) : plan_{other.plan_}, order_{other.order_} {}
  BasicEinsum &operator=(const BasicEinsum &other) {
    plan_ = other.plan_;
    order_ = other.order_;
    scratch_ = Scratch{};
    lowering_ = impl::LastLowering{};
    return *this;
  }
  BasicEinsum(BasicEinsum &&) noexcept = default;
  BasicEinsum &operator=(BasicEinsum &&) noexcept = default;
  ~BasicEinsum() = default;

  [[nodiscard]] constexpr const Subscripts &subscripts() const noexcept {
    return plan_.subscripts();
  }
  [[nodiscard]] constexpr std::size_t operand_count() const noexcept {
    return plan_.operand_count();
  }
  [[nodiscard]] constexpr const Labels &output_labels() const noexcept {
    return plan_.output_labels();
  }

  // obj(a, b, ...)           -> result<R>, by value, in the operands' own family
  // obj(a, b, ..., out(x))   -> result<void>, moved into x
  //
  // Which one a call is is settled by the tag and by nothing else, so an
  // operand needs no ceremony to stay one.
  template <typename... A> [[nodiscard]] auto operator()(A &&...args) const {
    if constexpr (impl::is_rt_out_form<A...>()) {
      return untag_output(std::forward_as_tuple(EINSUM_FWD(args)...),
                          std::make_index_sequence<sizeof...(A) - 1>{});
    } else {
      static_assert(CSameFamily<A...>,
                    "einsum(...): one call's operands must be one family and "
                    "one scalar -- all Eigen objects, all mdspans or spans, or "
                    "all nested ranges; their ranks may differ.  A trailing "
                    "non-const operand is read as an output, so pass operands "
                    "as const");
      if constexpr (!CSameFamily<A...>) {
        return result<void>{};
      } else {
        return by_value(args...);
      }
    }
  }

protected:
  BasicEinsum() = default;
  constexpr BasicEinsum(Plan plan, const path order) noexcept
      : plan_{std::move(plan)}, order_{order} {}
  friend struct impl::einsum_access;

  // Drop the tag and hand the storage it named to with_output, which wants the
  // output itself in the last slot.  Both objects come through here, so out()
  // means the same thing on either path.
  template <impl::CTupleLike Tup, std::size_t... I>
  result<void> untag_output(Tup tup, std::index_sequence<I...>) const {
    return with_output(std::forward_as_tuple(
                           std::get<I>(tup)...,
                           std::get<sizeof...(I)>(tup).target),
                       std::index_sequence<I...>{});
  }

  // The by-value form, deduced from bare types rather than from the forwarding
  // references above: result_of_t is written in terms of operand types, and a
  // deduced `T&` is not one.  The compile-time object calls this directly --
  // there the arity is a constant and has already settled the question, so it
  // must not be asked again.
  template <COperand... Ops>
    requires CSameFamily<Ops...>
  [[nodiscard]] result<result_of_t<Ops...>> by_value(const Ops &...ops) const {
    return evaluate<result_of_t<Ops...>, scalar_of_t<impl::first_of_t<Ops...>>>(
        ops...);
  }

  // Used by the compile-time object, where arity settles which form a call is.
  // R is the output's own type, not result_of_t of the operands: that is what
  // lets this form name a rank the by-value one cannot reach -- a rank-4 result
  // from rank-2 operands, say, which no operand type could have implied.
  template <impl::CTupleLike Tup, std::size_t... I>
  result<void> with_output(Tup tup, std::index_sequence<I...>) const {
    auto &out = std::get<sizeof...(I)>(tup);
    using R = std::remove_cvref_t<decltype(out)>;
    return evaluate<R, scalar_of_t<
                           impl::first_of_t<std::tuple_element_t<I, Tup>...>>>(
               std::get<I>(tup)...)
        .transform([&out](R &&value) noexcept { out = std::move(value); });
  }

  template <COperand R, CScalar T, COperand... Ops>
  [[nodiscard]] result<R> evaluate(const Ops &...ops) const;

  Plan plan_{};
  // Which contraction order to choose.  Part of what the object is, not of what
  // it caches: two objects over one subscript with different orders answer the
  // same values by different routes.
  path order_ = path::greedy;
  Scratch scratch_{};
  // Both caches are per-object and are not copied: a copy starts cold, and two
  // threads calling the same object share them, so that needs synchronising.
  mutable impl::LastLowering lowering_{};
};

// --- the one evaluation ------------------------------------------------------
// Shapes, then layouts, then the output shape the subscript implies, then the
// result object, then the geometry, then one scratch block, then execute.  Each
// step's failure is the call's, and every one of them is answered before a
// single element moves.
template <impl::CScratch Scratch>
template <COperand R, CScalar T, COperand... Ops>
result<R> BasicEinsum<Scratch>::evaluate(const Ops &...ops) const {
  constexpr std::size_t kOperands = sizeof...(Ops);
  if (kOperands != plan_.operand_count()) {
    return fail(errc::operand_count_mismatch);
  }

  const std::array<result<Shape>, kOperands> shapes{impl::shape_of(ops)...};
  for (const auto &shape : shapes) {
    if (!shape) {
      return propagate<R>(shape.error());
    }
  }

  impl::Layouts lays;
  std::size_t next = 0;
  (lays.push_back(impl::layout_for(ops, *shapes[next++])), ...);
  const std::span<const impl::Layout> spans{lays};

  // The lowering depends on the operand layouts and on nothing else, so a call
  // whose layouts match the last one's reuses what that produced.  Only the
  // description is cached; the contraction below always runs.
  constexpr bool kOutDirect = CDirectWritable<R>;
  const void *const kind = &impl::result_id<R>;
  if (!lowering_.matches(spans, kind)) {
    // A '...' stands for as many axes as the operands turn out to have, so the
    // subscript is not complete until they arrive.  Without one this expands to
    // itself; either way the plan that comes out is what the rest of the call
    // and the executor use.
    impl::FixedVec<std::uint8_t, kMaxOperands> ranks;
    for (const impl::Layout &lay : lays) {
      ranks.push_back(static_cast<std::uint8_t>(lay.rank()));
    }
    const auto expanded =
        impl::expand(plan_.subscripts(), std::span<const std::uint8_t>{ranks});
    if (!expanded) {
      return propagate<R>(expanded.error());
    }
    const auto fresh_plan = impl::make_plan(*expanded);
    if (!fresh_plan) {
      return propagate<R>(fresh_plan.error());
    }
    const auto fresh_shape = impl::infer_output_shape(*fresh_plan, spans);
    if (!fresh_shape) {
      return propagate<R>(fresh_shape.error());
    }
    const impl::Layout fresh_layout = impl::layout_of_result<R>(*fresh_shape);
    const auto fresh_geometry = impl::make_geometry(*fresh_plan, spans, fresh_layout, order_);
    if (!fresh_geometry) {
      return propagate<R>(fresh_geometry.error());
    }
    lowering_.operands.assign(lays.begin(), lays.end());
    lowering_.plan = *fresh_plan;
    lowering_.out_shape = *fresh_shape;
    lowering_.out_layout = fresh_layout;
    lowering_.geometry = *fresh_geometry;
    lowering_.result_kind = kind;
  }
  const Plan &plan = lowering_.plan;
  const Shape &out_shape = lowering_.out_shape;
  const impl::Layout &out_layout = lowering_.out_layout;
  const impl::Geometry &geometry = lowering_.geometry;

  auto out = impl::make_like<R>(out_shape);
  if (!out) {
    return propagate<R>(out.error());
  }
  // The result's own rank, which the shape above was fitted to; the elements
  // and their order are the same either way, so the executor is unaffected.
  const auto fitted = impl::fit_shape(
      out_shape, impl::CEigenDense<R> ? std::size_t{2} : rank_v<R>);
  if (!fitted) {
    return propagate<R>(fitted.error());
  }

  // One block, laid out by the same function the compile-time path uses; the
  // only difference here is that it is a pooled allocation rather than an array
  // in the frame, and that its size is not known until now.
  constexpr std::array<bool, kOperands> kGathered{!impl::CContiguous<Ops>...};
  const impl::ScratchMap map = impl::scratch_offsets(
      spans, std::span<const bool>{kGathered}, geometry.scratch_elems, out_shape,
      kOutDirect);

  const std::span<std::byte> block =
      scratch_.bytes(static_cast<std::size_t>(map.total) * sizeof(T));
  T *const pool = block.empty() ? nullptr : reinterpret_cast<T *>(block.data());

  impl::contract_into<T>(plan, geometry, spans, out_layout, pool, map, *out,
                         *fitted, ops...);
  return std::move(*out);
}

namespace impl {
struct einsum_access {
  template <CScratch Scratch>
  [[nodiscard]] static BasicEinsum<Scratch> make(Plan plan, const path order) noexcept {
    return BasicEinsum<Scratch>{std::move(plan), order};
  }

  // The order an object was built with, and the path its last call chose.  Both
  // are for tests: neither is on the public surface.
  template <CScratch Scratch>
  [[nodiscard]] static path order_of(const BasicEinsum<Scratch> &e) noexcept {
    return e.order_;
  }
  template <CScratch Scratch>
  [[nodiscard]] static const Path &last_path(const BasicEinsum<Scratch> &e) noexcept {
    return e.lowering_.geometry.path;
  }

  // What the runtime-rank entry point in <einsum/rt/dynamic.hpp> needs and a
  // caller has no business seeing: the lowering cache it shares with the typed
  // call operator, and the scratch block both draw from.
  template <CScratch Scratch>
  [[nodiscard]] static LastLowering &
  lowering_of(const BasicEinsum<Scratch> &e) noexcept {
    return e.lowering_;
  }
  template <CScratch Scratch>
  [[nodiscard]] static const Scratch &
  scratch_of(const BasicEinsum<Scratch> &e) noexcept {
    return e.scratch_;
  }

  // The compile-time counterpart: the path a StaticEinsum would choose for
  // these operands, which is a constant and so can be asserted rather than run.
  template <std::derived_from<BasicEinsum<HeapScratch>> E, COperand... Ops>
  [[nodiscard]] static consteval Path static_path() noexcept {
    return E::template Lowered<Ops...>::geometry.path;
  }
};
} // namespace impl

using Einsum = BasicEinsum<impl::HeapScratch>;

} // namespace einsum
