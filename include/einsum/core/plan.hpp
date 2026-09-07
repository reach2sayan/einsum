#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <string_view>

namespace einsum {

// What one operand needs before it can take part in a contraction: its repeated
// labels merged onto a diagonal, and the labels nobody else will ever see
// summed away.  Both are cheaper here than inside the GEMM loop, and both are
// what make the step classification below total.
struct Prep {
  // The merged axes, one per distinct label, first occurrence order.
  Labels merged_labels{};
  // Per original axis, which merged axis it folds into.  Two axes sharing one
  // means a diagonal, and the merged stride is the sum of theirs.
  impl::FixedVec<std::uint8_t, kMaxRank> merged_into{};
  // Merged labels no other operand has and the output does not want.
  Labels reduce{};
  // merged_labels minus reduce: what the steps actually see.
  Labels labels_after{};

  [[nodiscard]] friend constexpr bool
  operator==(const Prep &, const Prep &) noexcept = default;
};

// One binary contraction, named by which labels play which role in it.  batch
// is the pair of axes GEMM iterates over, m/n are the free axes of the two
// sides, k is what is summed.  Nothing here mentions an extent: a Plan is the
// shape of the computation, not of the data.
// Where a step's two sides come from: an operand by index, or the result of an
// earlier step, which is kIntermediate plus that step's index.  A path chosen by
// cost does not contract left to right, so a step has to say.
inline constexpr std::uint8_t kIntermediate = kMaxOperands;

struct Step {
  Labels batch{};
  Labels m{};
  Labels n{};
  Labels k{};
  Labels target{}; // batch ++ m ++ n, the labels of what this step produces
  std::uint8_t l_src = 0;
  std::uint8_t r_src = 0;
  bool writes_output = false;

  [[nodiscard]] friend constexpr bool
  operator==(const Step &, const Step &) noexcept = default;
};

// What each label is bound to, and whether anything has bound it yet.  Here
// rather than in the lowering because a path policy is chosen on extents and
// must be able to read them.
struct BoundExtent {
  index_t extent = 0;
  bool known = false;
  bool broadcast = false;

  [[nodiscard]] friend constexpr bool operator==(const BoundExtent &,
                                                 const BoundExtent &) noexcept = default;
};

using BoundExtents = impl::LabelTable<BoundExtent>;

class Plan;

namespace impl {
[[nodiscard]] constexpr result<Plan> make_plan(const Subscripts &) noexcept;

// The one way the lowering reads a Plan's internals.  A friend struct rather
// than a public accessor per member: preps and steps are what make_geometry
// walks and are no part of what a caller of einsum() ever asks about.
struct access;
} // namespace impl

// A parsed, lowered einsum: extent-free, allocation-free, a value type.  It is
// the same object whichever parser built it, so a plan can be checked at
// compile time and executed at run time.
class Plan {
public:
  constexpr Plan() noexcept = default;

  [[nodiscard]] constexpr const Subscripts &subscripts() const noexcept {
    return subs_;
  }
  [[nodiscard]] constexpr std::size_t operand_count() const noexcept {
    return subs_.operands.size();
  }
  [[nodiscard]] constexpr const Labels &output_labels() const noexcept {
    return subs_.output;
  }
  [[nodiscard]] friend constexpr bool
  operator==(const Plan &, const Plan &) noexcept = default;

private:
  friend constexpr result<Plan> impl::make_plan(const Subscripts &) noexcept;
  friend struct impl::access;

  Subscripts subs_{};
  impl::FixedVec<Prep, kMaxOperands> preps_{};
};

namespace impl {

struct access {
  [[nodiscard]] static constexpr const FixedVec<Prep, kMaxOperands> &
  preps(const Plan &plan) noexcept {
    return plan.preps_;
  }
};

[[nodiscard]] constexpr result<Prep>
make_prep(const Labels &operand, const Labels &output,
          const LabelTable<std::uint8_t> &counts) noexcept {
  Prep prep;
  for (const char c : operand) {
    const std::size_t at = index_of(prep.merged_labels, c);
    if (at == prep.merged_labels.size() &&
        !prep.merged_labels.try_push_back(c)) {
      return fail(errc::rank_too_high);
    }
    if (!prep.merged_into.try_push_back(static_cast<std::uint8_t>(at))) {
      return fail(errc::rank_too_high);
    }
  }
  for (const char c : prep.merged_labels) {
    const bool nobody_else = counts[c] == 1;
    if (nobody_else && !std::ranges::contains(output, c)) {
      prep.reduce.push_back(c);
    } else {
      prep.labels_after.push_back(c);
    }
  }
  return prep;
}

// The subscript and what each operand must do to itself before it can take part
// -- the diagonals it walks and the axes it sums away.  In what ORDER the
// operands are then contracted is not decided here: that depends on their
// extents, which a Plan has never seen, so it belongs to the per-call lowering
// and to the path policy that drives it.
[[nodiscard]] constexpr result<Plan>
make_plan(const Subscripts &subs) noexcept {
  Plan plan;
  plan.subs_ = subs;

  // Once per operand: "ii" is one occurrence of i and a diagonal, not two
  // occurrences and a contraction.
  const auto counts = count_labels(subs.operands, true);
  for (const Labels &op : subs.operands) {
    const auto prep = make_prep(op, subs.output, counts);
    if (!prep) {
      return std::unexpected{prep.error()};
    }
    plan.preps_.push_back(*prep);
  }

  return plan;
}

// parse -> plan, as one pipeline both front ends run.  ct::subscripts<S> calls
// it in a constant expression and rt::plan() calls it after Boost.Parser has
// produced the same Subscripts, so the two paths differ only in the parser.
[[nodiscard]] constexpr result<Plan>
build_plan(const std::string_view source) noexcept {
  return parse_subscripts(source).and_then(make_plan);
}

} // namespace impl

} // namespace einsum
