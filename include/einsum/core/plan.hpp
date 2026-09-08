#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <string_view>

namespace einsum {

// What one operand does to itself before contracting: repeated labels merged
// onto a diagonal, private labels summed away.  Both are cheaper here than
// inside the GEMM loop.
struct Prep {
  Labels merged_labels{}; // one per distinct label, first occurrence order
  // Per original axis, which merged axis it folds into; two axes sharing one
  // are a diagonal, whose stride is the sum of theirs.
  impl::FixedVec<std::uint8_t, kMaxRank> merged_into{};
  Labels reduce{};       // merged labels no other operand has and the output does not want
  Labels labels_after{}; // merged_labels minus reduce: what the steps see

  [[nodiscard]] friend constexpr bool
  operator==(const Prep &, const Prep &) noexcept = default;
};

// A step's side: an operand by index, or this plus an earlier step's index.
inline constexpr std::uint8_t kIntermediate = kMaxOperands;

// One binary contraction: batch is what GEMM iterates over, m/n are the two
// sides' free axes, k is what is summed.  No extents here.
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

// Here rather than in the lowering: a path is chosen on extents.
struct BoundExtent {
  index_t extent = 0;
  bool known = false;

  [[nodiscard]] friend constexpr bool
  operator==(const BoundExtent &, const BoundExtent &) noexcept = default;
};

using BoundExtents = impl::LabelTable<BoundExtent>;

class Plan;

namespace impl {
[[nodiscard]] constexpr result<Plan> make_plan(const Subscripts &) noexcept;

// The one way the lowering reads a Plan's internals.
struct access;
} // namespace impl

// Extent-free, allocation-free, a value type, and the same whichever parser
// built it.
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
  std::ranges::partition_copy(
      prep.merged_labels, std::back_inserter(prep.reduce),
      std::back_inserter(prep.labels_after), [&](const char c) {
        return counts[c] == 1 && !std::ranges::contains(output, c);
      });
  return prep;
}

// The order the operands are then contracted in depends on their extents,
// which a Plan has never seen, so it belongs to the per-call lowering.
[[nodiscard]] constexpr result<Plan>
make_plan(const Subscripts &subs) noexcept {
  Plan plan;
  plan.subs_ = subs;

  // Once per operand: "ii" is one occurrence of i and a diagonal.
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

// parse -> plan, as one pipeline both front ends run.
[[nodiscard]] constexpr result<Plan>
build_plan(const std::string_view source) noexcept {
  return parse_subscripts(source).and_then(make_plan);
}

} // namespace impl

} // namespace einsum
