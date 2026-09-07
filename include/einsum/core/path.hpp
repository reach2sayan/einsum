#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ranges>
#include <span>

// In what order the operands are contracted.  The subscript says WHAT is
// summed; this says in which order the binary contractions happen, which the
// subscript does not constrain and which changes the cost by orders of
// magnitude: "ij,jk,kl->il" over a thin middle is cheap left to right and dear
// right to left, and the reverse holds when the middle is fat.
//
// The choice is a value, not a policy type: the runtime object stores it and
// the lowering switches on it, so einsum() stays one non-template type.
namespace einsum {

// Which order to choose.  greedy is the default because it is the one that is
// right when the caller has not thought about it.
enum class path : std::uint8_t { greedy, sequential };

// The chosen order.  A FixedVec rather than a runtime container because a
// Geometry has to survive into a constant expression on the compile-time path.
struct Path {
  impl::FixedVec<Step, kMaxOperands - 1> steps{};

  [[nodiscard]] friend constexpr bool operator==(const Path &, const Path &) noexcept = default;
};

namespace impl {

// One tensor still waiting to be contracted: what labels it carries and where
// it came from.
struct Live {
  Labels labels{};
  std::uint8_t src = 0;
};

// The extent a label is bound to, or 1 for one nothing has bound -- which is
// the identity for the products below, so an unbound label costs nothing.
[[nodiscard]] constexpr index_t extent_of(const BoundExtents &bound, const char c) noexcept {
  return bound[c].known ? bound[c].extent : 1;
}

[[nodiscard]] constexpr index_t volume(const Labels &labels, const BoundExtents &bound) noexcept {
  return std::ranges::fold_left(labels, index_t{1}, [&](const index_t acc, const char c) {
    return acc * extent_of(bound, c);
  });
}

// The one classification, shared by every policy: of the labels the two sides
// carry, those both have and nothing else needs are summed (k), those both have
// and something still wants are iterated (batch), and the rest are free (m from
// the left, n from the right).  `rest` is every other live tensor, which is
// what "something still wants" means once the order is no longer left to right.
[[nodiscard]] constexpr result<Step> make_step(const Labels &left, const Labels &right,
                                               const Labels &output,
                                               const std::span<const Live> rest) noexcept {
  const auto needed = [&](const char c) noexcept {
    return std::ranges::contains(output, c) ||
           std::ranges::any_of(rest, [c](const Live &other) {
             return std::ranges::contains(other.labels, c);
           });
  };

  Step step;
  for (const char c : left) {
    const bool shared = std::ranges::contains(right, c);
    Labels &group = shared ? (needed(c) ? step.batch : step.k) : step.m;
    if (!group.try_push_back(c)) {
      return fail(errc::rank_too_high);
    }
  }
  // Right-only labels are all free: one that nothing else wants would have been
  // summed away by make_prep, since only that operand could have had it.
  for (const char c : right) {
    if (!std::ranges::contains(left, c) && !step.n.try_push_back(c)) {
      return fail(errc::rank_too_high);
    }
  }
  for (const Labels *group : {&step.batch, &step.m, &step.n}) {
    for (const char c : *group) {
      if (!step.target.try_push_back(c)) {
        return fail(errc::rank_too_high);
      }
    }
  }
  return step;
}

// Everything live except the two being contracted, which is what make_step has
// to be told to classify a pair correctly.
[[nodiscard]] constexpr FixedVec<Live, kMaxOperands>
others(const FixedVec<Live, kMaxOperands> &live, const std::size_t a,
       const std::size_t b) noexcept {
  FixedVec<Live, kMaxOperands> rest;
  for (const auto i : std::views::iota(std::size_t{0}, live.size())) {
    if (i != a && i != b) {
      rest.push_back(live[i]);
    }
  }
  return rest;
}

// Contract `a` with `b`, record the step, and replace the pair by its result.
[[nodiscard]] constexpr result<void> take(FixedVec<Live, kMaxOperands> &live, Path &into,
                                          const Labels &output, const std::size_t a,
                                          const std::size_t b) noexcept {
  const auto rest = others(live, a, b);
  auto step = make_step(live[a].labels, live[b].labels, output, std::span<const Live>{rest});
  if (!step) {
    return std::unexpected{step.error()};
  }
  step->l_src = live[a].src;
  step->r_src = live[b].src;

  const Live made{.labels = step->target,
                  .src = static_cast<std::uint8_t>(kIntermediate + into.steps.size())};
  if (!into.steps.try_push_back(*step)) {
    return fail(errc::too_many_operands);
  }

  // The new tensor takes the front, which is what makes `sequential` accumulate
  // to the left the way the subscript reads: contracting (0, 1) each turn then
  // means "what I have so far, with the next operand".  For a cost-chosen path
  // the position is immaterial.
  FixedVec<Live, kMaxOperands> next;
  next.push_back(made);
  for (const auto i : std::views::iota(std::size_t{0}, live.size())) {
    if (i != a && i != b) {
      next.push_back(live[i]);
    }
  }
  live = next;
  return {};
}

[[nodiscard]] constexpr FixedVec<Live, kMaxOperands>
live_from(const std::span<const Labels> operands) noexcept {
  FixedVec<Live, kMaxOperands> live;
  for (const auto i : std::views::iota(std::size_t{0}, operands.size())) {
    live.push_back({.labels = operands[i], .src = static_cast<std::uint8_t>(i)});
  }
  return live;
}

// The last step is the one that writes the caller's output.
constexpr void mark_last(Path &into) noexcept {
  if (!into.steps.empty()) {
    into.steps[into.steps.size() - 1].writes_output = true;
  }
}

} // namespace impl

namespace impl {

// The subscript's own order, left to right, no reassociation.  What the library
// did before there was a choice, kept so that a caller who has tuned a
// subscript by hand still gets what they wrote.
[[nodiscard]] constexpr result<Path> sequential_path(const std::span<const Labels> operands,
                                                     const Labels &output) noexcept {
  Path chosen;
  auto live = live_from(operands);
  while (live.size() > 1) {
    if (const auto ok = take(live, chosen, output, 0, 1); !ok) {
      return std::unexpected{ok.error()};
    }
  }
  mark_last(chosen);
  return chosen;
}

// opt_einsum's greedy: at each turn contract the pair whose intermediate is
// cheapest to form, breaking ties towards the pair that sums the most away.
// Not optimal -- that is exponential -- but it is what turns an "ij,jk,kl->il"
// with a thin middle from a dense square into two thin products.
[[nodiscard]] constexpr result<Path> greedy_path(const std::span<const Labels> operands,
                                                 const Labels &output,
                                                 const BoundExtents &bound) noexcept {
  Path chosen;
  auto live = live_from(operands);
  while (live.size() > 1) {
    std::size_t best_a = 0;
    std::size_t best_b = 1;
    index_t best_cost = 0;
    index_t best_removed = 0;
    bool first = true;

    for (const auto a : std::views::iota(std::size_t{0}, live.size())) {
      for (const auto b : std::views::iota(a + 1, live.size())) {
        const auto rest = others(live, a, b);
        const auto step =
            make_step(live[a].labels, live[b].labels, output, std::span<const Live>{rest});
        if (!step) {
          return std::unexpected{step.error()};
        }
        // What the pair costs is the tensor it makes; what it earns is the
        // labels it sums away.
        const index_t cost = volume(step->target, bound);
        const index_t removed = volume(step->k, bound);
        if (first || cost < best_cost || (cost == best_cost && removed > best_removed)) {
          best_a = a;
          best_b = b;
          best_cost = cost;
          best_removed = removed;
          first = false;
        }
      }
    }
    if (const auto ok = take(live, chosen, output, best_a, best_b); !ok) {
      return std::unexpected{ok.error()};
    }
  }
  mark_last(chosen);
  return chosen;
}

// The one entry point the lowering calls.  Extents rather than raw shapes, and
// the operands' labels after their diagonals and reductions rather than the
// Subscripts: both are what the lowering already has in hand, and recomputing
// either here would be a second place for them to be got wrong.
[[nodiscard]] constexpr result<Path> choose_path(const path which,
                                                 const std::span<const Labels> operands,
                                                 const Labels &output,
                                                 const BoundExtents &bound) noexcept {
  return which == path::sequential ? sequential_path(operands, output)
                                   : greedy_path(operands, output, bound);
}

} // namespace impl

namespace impl {
// What a path costs, for the tests and the README table.  The sum over steps of
// the product of every label the step touches, which is the usual count.
[[nodiscard]] constexpr index_t flops(const Path &chosen, const BoundExtents &bound) noexcept {
  return std::ranges::fold_left(chosen.steps, index_t{0}, [&](const index_t acc, const Step &s) {
    return acc + volume(s.target, bound) * volume(s.k, bound);
  });
}
} // namespace impl

} // namespace einsum
