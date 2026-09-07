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

#include <boost/container/static_vector.hpp>

#include <array>
#include <cstddef>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsum {

namespace impl {

// --- scratch policies
// --------------------------------------------------------- The object is
// immutable: nothing a caller can observe changes between calls. The scratch is
// therefore not state but a cache, which is what `mutable` is for -- and why
// both call operators can be const.  It grows to the high-water mark of the
// calls made through this object and is never shrunk.
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

// --- the geometry cache -------------------------------------------------------
// The lowering depends on the operands' shapes AND strides and on nothing else,
// so a call whose layouts match the last one's can reuse the Geometry it
// produced.  Like the scratch this is a cache, not state: it changes nothing a
// caller can observe, which is why the call operators stay const.
//
// Values are never compared and never kept -- only layouts -- so the
// contraction itself always runs.
// One address per result type, so a record made for an Eigen result is never
// handed to a call that wants an mdarray -- two operand layouts can be equal
// while the results they imply are not.
template <typename R> inline constexpr char result_id = 0;

struct LastLowering {
  boost::container::static_vector<Layout, kMaxOperands> operands{};
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
template <typename R>
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
template <typename Tup, std::size_t... I>
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

} // namespace impl

// --- the object
// --------------------------------------------------------------- A parsed
// subscript and nothing else a caller can see.  Both call operators are const:
// the object is immutable, so one may be shared, stored by value, or made
// constexpr on the compile-time path.  Concurrent calls on the *same* object
// still need synchronisation, because they share the scratch cache.
template <typename Scratch> class BasicEinsum;

namespace impl {
// The one way an Einsum is built from a Plan.  A caller never has a Plan --
// einsum() hands back the finished object -- so the constructor that takes one
// is no part of the surface.
struct einsum_access;
} // namespace impl

template <typename Scratch> class BasicEinsum {
public:
  // A copy starts cold: the caches belong to the object that warmed them, and
  // sharing them would make two objects that must not interact do so.
  BasicEinsum(const BasicEinsum &other) : plan_{other.plan_} {}
  BasicEinsum &operator=(const BasicEinsum &other) {
    plan_ = other.plan_;
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

  // obj(a, b, ...): the result, by value, in the operands' own family.
  //
  // There is no output-parameter form here, and deliberately: the operand count
  // is a run-time value on this path, so a trailing non-const matrix could not
  // be told from one more operand, and `out = *e(a, b)` is already a move for
  // every family this returns.
  template <COperand... Ops>
    requires CSameFamily<Ops...>
  [[nodiscard]] result<result_of_t<Ops...>>
  operator()(const Ops &...ops) const {
    return evaluate<result_of_t<Ops...>, scalar_of_t<impl::first_of_t<Ops...>>>(
        ops...);
  }

protected:
  BasicEinsum() = default;
  constexpr explicit BasicEinsum(Plan plan) noexcept : plan_{std::move(plan)} {}
  friend struct impl::einsum_access;

  // Used by the compile-time object, where arity settles which form a call is.
  // R is the output's own type, not result_of_t of the operands: that is what
  // lets this form name a rank the by-value one cannot reach -- a rank-4 result
  // from rank-2 operands, say, which no operand type could have implied.
  template <typename Tup, std::size_t... I>
  result<void> with_output(Tup tup, std::index_sequence<I...>) const {
    auto &out = std::get<sizeof...(I)>(tup);
    using R = std::remove_cvref_t<decltype(out)>;
    return evaluate<R, scalar_of_t<
                           impl::first_of_t<std::tuple_element_t<I, Tup>...>>>(
               std::get<I>(tup)...)
        .transform([&out](R &&value) noexcept { out = std::move(value); });
  }

  template <typename R, CScalar T, COperand... Ops>
  [[nodiscard]] result<R> evaluate(const Ops &...ops) const;

  Plan plan_{};
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
template <typename Scratch>
template <typename R, CScalar T, COperand... Ops>
result<R> BasicEinsum<Scratch>::evaluate(const Ops &...ops) const {
  constexpr std::size_t kOperands = sizeof...(Ops);
  if (kOperands != plan_.operand_count()) {
    return fail(errc::operand_count_mismatch);
  }

  const std::array<result<Shape>, kOperands> shapes{shape_of(ops)...};
  for (const auto &shape : shapes) {
    if (!shape) {
      return propagate<R>(shape.error());
    }
  }

  impl::Layouts lays;
  std::size_t next = 0;
  (lays.push_back(impl::layout_for(ops, *shapes[next++])), ...);
  const std::span<const Layout> spans{lays};

  // The lowering depends on the operand layouts and on nothing else, so a call
  // whose layouts match the last one's reuses what that produced.  Only the
  // description is cached; the contraction below always runs.
  constexpr bool kOutDirect = impl::CEigenDense<R> || impl::CEigenTensor<R> ||
                              impl::CMdarray<R> || rank_v<R> == 1;
  const void *const kind = &impl::result_id<R>;
  if (!lowering_.matches(spans, kind)) {
    const auto fresh_shape = impl::infer_output_shape(plan_, spans);
    if (!fresh_shape) {
      return propagate<R>(fresh_shape.error());
    }
    const Layout fresh_layout = impl::layout_of_result<R>(*fresh_shape);
    const auto fresh_geometry = impl::make_geometry(plan_, spans, fresh_layout);
    if (!fresh_geometry) {
      return propagate<R>(fresh_geometry.error());
    }
    lowering_.operands.assign(lays.begin(), lays.end());
    lowering_.out_shape = *fresh_shape;
    lowering_.out_layout = fresh_layout;
    lowering_.geometry = *fresh_geometry;
    lowering_.result_kind = kind;
  }
  const Shape &out_shape = lowering_.out_shape;
  const Layout &out_layout = lowering_.out_layout;
  const impl::Geometry &geometry = lowering_.geometry;

  auto out = impl::make_like<R>(out_shape);
  if (!out) {
    return propagate<R>(out.error());
  }
  // The result's own rank, which the shape above was fitted to; the elements
  // and their order are the same either way, so the executor is unaffected.
  const auto fitted =
      impl::fit_shape(out_shape, impl::CEigenDense<R> ? std::size_t{2} : rank_v<R>);
  if (!fitted) {
    return propagate<R>(fitted.error());
  }

  // One block: the geometry's own scratch first, because its offsets are
  // relative to what execute() is handed, then a packed copy of every operand
  // the kernels cannot address, then the output when it is one of them.
  const auto elems = static_cast<std::size_t>(geometry.scratch_elems);
  auto cursor = static_cast<index_t>(elems);
  std::array<index_t, kOperands> packed_at{};
  next = 0;
  const auto reserve = [&](const auto &op) {
    if constexpr (CContiguous<decltype(op)>) {
      packed_at[next] = -1;
    } else {
      packed_at[next] = cursor;
      cursor += impl::product(*shapes[next]);
    }
    ++next;
  };
  (reserve(ops), ...);
  const index_t out_at = kOutDirect ? -1 : cursor;
  if constexpr (!kOutDirect) {
    cursor += impl::product(out_shape);
  }

  const std::span<std::byte> block =
      scratch_.bytes(static_cast<std::size_t>(cursor) * sizeof(T));
  T *const pool = block.empty() ? nullptr : reinterpret_cast<T *>(block.data());

  boost::container::static_vector<TensorView<const T>, kMaxOperands> views;
  next = 0;
  const auto prepare = [&](const auto &op) {
    const T *base = nullptr;
    if constexpr (CContiguous<decltype(op)>) {
      base = impl::data_of(op);
    } else {
      T *const packed = pool + packed_at[next];
      gather(op, packed, *shapes[next]);
      base = packed;
    }
    views.push_back(TensorView<const T>{base, lays[next]});
    ++next;
  };
  (prepare(ops), ...);

  // if constexpr, not a ternary: the two branches have different pointer types.
  T *target_data = nullptr;
  if constexpr (kOutDirect) {
    target_data = out->data();
  } else {
    target_data = pool + out_at;
  }
  const TensorView<T> target{target_data, out_layout};
  impl::execute<T>(plan_, geometry, std::span<const TensorView<const T>>{views},
                   target, std::span<T>{pool, elems});

  if constexpr (!kOutDirect) {
    scatter(static_cast<const T *>(pool + out_at), *out, *fitted);
  }
  return std::move(*out);
}

namespace impl {
struct einsum_access {
  template <typename Scratch>
  [[nodiscard]] static BasicEinsum<Scratch> make(Plan plan) noexcept {
    return BasicEinsum<Scratch>{std::move(plan)};
  }
};
} // namespace impl

using Einsum = BasicEinsum<impl::HeapScratch>;

} // namespace einsum
