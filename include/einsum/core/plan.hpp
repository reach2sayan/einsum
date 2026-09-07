#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"

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

  [[nodiscard]] friend constexpr bool operator==(const Prep &,
                                                 const Prep &) noexcept = default;
};

// One binary contraction, named by which labels play which role in it.  batch
// is the pair of axes GEMM iterates over, m/n are the free axes of the two
// sides, k is what is summed.  Nothing here mentions an extent: a Plan is the
// shape of the computation, not of the data.
struct Step {
  Labels batch{};
  Labels m{};
  Labels n{};
  Labels k{};
  Labels target{}; // batch ++ m ++ n, the labels of what this step produces
  bool writes_output = false;

  [[nodiscard]] friend constexpr bool operator==(const Step &,
                                                 const Step &) noexcept = default;
};

class Plan;

namespace impl {
[[nodiscard]] constexpr result<Plan> make_plan(const Subscripts &) noexcept;
} // namespace impl

// A parsed, lowered einsum: extent-free, allocation-free, a value type.  It is
// the same object whichever parser built it, so a plan can be checked at
// compile time and executed at run time.
class Plan {
public:
  constexpr Plan() noexcept = default;

  [[nodiscard]] constexpr const Subscripts &subscripts() const noexcept { return subs_; }
  [[nodiscard]] constexpr std::size_t operand_count() const noexcept {
    return subs_.operands.size();
  }
  [[nodiscard]] constexpr const Labels &output_labels() const noexcept {
    return subs_.output;
  }
  [[nodiscard]] constexpr const impl::FixedVec<Prep, kMaxOperands> &preps() const noexcept {
    return preps_;
  }
  [[nodiscard]] constexpr const impl::FixedVec<Step, kMaxOperands - 1> &
  steps() const noexcept {
    return steps_;
  }

  [[nodiscard]] friend constexpr bool operator==(const Plan &,
                                                 const Plan &) noexcept = default;

  // Defined in core/bound.hpp, which is where Bound<T> becomes a complete type.
  // Templates, so nothing is instantiated until a caller names one.
  template <typename... A> [[nodiscard]] auto bind(A &&...args) const;
  template <typename... A> [[nodiscard]] auto operator()(A &&...args) const;
  template <typename... A> [[nodiscard]] auto eval(A &&...args) const;
  template <typename... A> [[nodiscard]] auto to_matrix(A &&...args) const;

private:
  friend constexpr result<Plan> impl::make_plan(const Subscripts &) noexcept;

  Subscripts subs_{};
  impl::FixedVec<Prep, kMaxOperands> preps_{};
  impl::FixedVec<Step, kMaxOperands - 1> steps_{};
};

namespace impl {

[[nodiscard]] constexpr result<Prep>
make_prep(const Labels &operand, const Labels &output,
          const LabelTable<std::uint8_t> &counts) noexcept {
  Prep prep;
  for (const char c : operand) {
    const std::size_t at = prep.merged_labels.index_of(c);
    if (at == prep.merged_labels.size() && !prep.merged_labels.push_back(c)) {
      return fail(errc::rank_too_high);
    }
    if (!prep.merged_into.push_back(static_cast<std::uint8_t>(at))) {
      return fail(errc::rank_too_high);
    }
  }
  for (const char c : prep.merged_labels) {
    const bool nobody_else = counts[c] == 1;
    if (nobody_else && !output.contains(c)) {
      (void)prep.reduce.push_back(c);
    } else {
      (void)prep.labels_after.push_back(c);
    }
  }
  return prep;
}

// Left to right, no reassociation: the subscript's order is the caller's
// choice of contraction order, and a plan that reordered it would be optimising
// a cost model this library does not have.
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
    (void)plan.preps_.push_back(*prep);
  }

  const std::size_t n = plan.preps_.size();
  Labels left = plan.preps_[0].labels_after;

  for (const auto i : std::views::iota(std::size_t{1}, n)) {
    const Labels &right = plan.preps_[i].labels_after;

    // A label is needed past this step if the output wants it or a later
    // operand still has to meet it; anything else is what k means.
    const auto needed = [&](const char c) noexcept {
      if (subs.output.contains(c)) {
        return true;
      }
      return std::ranges::any_of(
          std::views::iota(i + 1, n),
          [&](const std::size_t j) { return plan.preps_[j].labels_after.contains(c); });
    };

    Step step;
    for (const char c : left) {
      const bool shared = right.contains(c);
      auto &group = shared ? (needed(c) ? step.batch : step.k) : step.m;
      if (!group.push_back(c)) {
        return fail(errc::rank_too_high);
      }
    }
    // Right-only labels are all free: one that nothing later wants would have
    // been summed away by make_prep, since only this operand could have it.
    for (const char c : right) {
      if (!left.contains(c) && !step.n.push_back(c)) {
        return fail(errc::rank_too_high);
      }
    }

    for (const Labels *group : {&step.batch, &step.m, &step.n}) {
      for (const char c : *group) {
        if (!step.target.push_back(c)) {
          return fail(errc::rank_too_high);
        }
      }
    }
    step.writes_output = (i + 1 == n);
    left = step.target;
    (void)plan.steps_.push_back(step);
  }

  return plan;
}

// parse -> plan, as one pipeline both front ends run.  ct::subscripts<S> calls
// it in a constant expression and rt::plan() calls it after Boost.Parser has
// produced the same Subscripts, so the two paths differ only in the parser.
[[nodiscard]] constexpr result<Plan> build_plan(const std::string_view source) noexcept {
  return parse_subscripts(source).and_then(make_plan);
}

} // namespace impl

} // namespace einsum
