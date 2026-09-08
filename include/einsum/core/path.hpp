#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <utility>

// In what order the binary contractions happen, which the subscript does not
// constrain and which changes the cost by orders of magnitude.  A value, not a
// policy type, so einsum() stays one non-template type.
namespace einsum {

enum class path : std::uint8_t { greedy, sequential };

// A FixedVec because a Geometry has to survive into a constant expression.
struct Path {
  impl::FixedVec<Step, kMaxOperands - 1> steps{};

  [[nodiscard]] friend constexpr bool
  operator==(const Path &, const Path &) noexcept = default;
};

namespace impl {

// One tensor still waiting to be contracted.
struct Live {
  Labels labels{};
  std::uint8_t src = 0;
};

// 1 for an unbound label: the identity of the products below.
[[nodiscard]] constexpr index_t extent_of(const BoundExtents &bound,
                                          const char c) noexcept {
  return bound[c].known ? bound[c].extent : 1;
}

[[nodiscard]] constexpr index_t volume(const Labels &labels,
                                       const BoundExtents &bound) noexcept {
  return product(labels | std::views::transform(
                              [&](const char c) { return extent_of(bound, c); }));
}

// The one classification, shared by every policy: labels both sides have and
// nothing else needs are summed (k), those something still wants are iterated
// (batch), the rest are free (m left, n right).  `rest` -- every other live
// tensor -- is what "something still wants" means.
[[nodiscard]] constexpr result<Step>
make_step(const Labels &left, const Labels &right, const Labels &output,
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
  // Right-only labels are all free: one nothing else wants was summed away by
  // make_prep already.
  for (const char c : right) {
    if (!std::ranges::contains(left, c) && !step.n.try_push_back(c)) {
      return fail(errc::rank_too_high);
    }
  }
  for (const char c : std::array<std::span<const char>, 3>{
           step.batch, step.m, step.n} |
                          std::views::join) {
    if (!step.target.try_push_back(c)) {
      return fail(errc::rank_too_high);
    }
  }
  return step;
}

// Everything live except the two being contracted, which is what make_step
// needs to classify a pair.
[[nodiscard]] constexpr FixedVec<Live, kMaxOperands>
others(const FixedVec<Live, kMaxOperands> &live, const std::size_t a,
       const std::size_t b) noexcept {
  return {std::from_range,
          std::views::iota(std::size_t{0}, live.size()) |
              std::views::filter(
                  [=](const std::size_t i) { return i != a && i != b; }) |
              std::views::transform(
                  [&](const std::size_t i) { return live[i]; })};
}

// Contract `a` with `b`, record the step, and replace the pair by its result.
[[nodiscard]] constexpr result<void> take(FixedVec<Live, kMaxOperands> &live,
                                          Path &into, const Labels &output,
                                          const std::size_t a,
                                          const std::size_t b) noexcept {
  const auto rest = others(live, a, b);
  auto step = make_step(live[a].labels, live[b].labels, output,
                        std::span<const Live>{rest});
  if (!step) {
    return std::unexpected{step.error()};
  }
  step->l_src = live[a].src;
  step->r_src = live[b].src;

  const Live made{
      .labels = step->target,
      .src = static_cast<std::uint8_t>(kIntermediate + into.steps.size())};
  if (!into.steps.try_push_back(*step)) {
    return fail(errc::too_many_operands);
  }

  // The new tensor takes the front, which is what makes `sequential` accumulate
  // leftwards.  For a cost-chosen path the position is immaterial.
  live.clear();
  live.push_back(made);
  std::ranges::copy(rest, std::back_inserter(live));
  return {};
}

[[nodiscard]] constexpr FixedVec<Live, kMaxOperands>
live_from(const std::span<const Labels> operands) noexcept {
  return {std::from_range,
          operands | std::views::enumerate |
              std::views::transform([](const auto &pair) {
                const auto &[i, labels] = pair;
                return Live{.labels = labels,
                            .src = static_cast<std::uint8_t>(i)};
              })};
}

// The last step is the one that writes the caller's output.
constexpr void mark_last(Path &into) noexcept {
  if (!into.steps.empty()) {
    into.steps.back().writes_output = true;
  }
}

// The subscript's own order, left to right, no reassociation.
[[nodiscard]] constexpr result<Path>
sequential_path(const std::span<const Labels> operands,
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

// The pairs of a set of that size, each once, as one flat range.
[[nodiscard]] constexpr auto pairs_of(const std::size_t n) noexcept {
  return std::views::iota(std::size_t{0}, n) |
         std::views::transform([n](const std::size_t a) {
           return std::views::iota(a + 1, n) |
                  std::views::transform([a](const std::size_t b) {
                    return std::pair{a, b};
                  });
         }) |
         std::views::join;
}

// opt_einsum's greedy: contract the pair whose intermediate is cheapest to
// form, ties to the pair that sums the most away.  Not optimal -- that is
// exponential -- but it turns "ij,jk,kl->il" over a thin middle into two thin
// products rather than a dense square.
[[nodiscard]] constexpr result<Path>
greedy_path(const std::span<const Labels> operands, const Labels &output,
            const BoundExtents &bound) noexcept {
  Path chosen;
  auto live = live_from(operands);
  while (live.size() > 1) {
    std::pair best{std::size_t{0}, std::size_t{1}};
    // Cost is the tensor the pair makes, and what it earns is what it sums
    // away.
    std::pair best_key{std::numeric_limits<index_t>::max(), index_t{0}};

    for (const auto [a, b] : pairs_of(live.size())) {
      const auto rest = others(live, a, b);
      const auto step = make_step(live[a].labels, live[b].labels, output,
                                  std::span<const Live>{rest});
      if (!step) {
        return std::unexpected{step.error()};
      }
      if (const std::pair key{volume(step->target, bound),
                              -volume(step->k, bound)};
          key < best_key) {
        best_key = key;
        best = {a, b};
      }
    }
    if (const auto ok = take(live, chosen, output, best.first, best.second);
        !ok) {
      return std::unexpected{ok.error()};
    }
  }
  mark_last(chosen);
  return chosen;
}

// The one entry point the lowering calls: bound extents and the labels left
// after the diagonals and reductions, both of which it already has in hand.
[[nodiscard]] constexpr result<Path>
choose_path(const path which, const std::span<const Labels> operands,
            const Labels &output, const BoundExtents &bound) noexcept {
  return which == path::sequential ? sequential_path(operands, output)
                                   : greedy_path(operands, output, bound);
}

// What a path costs, for the tests and the README table.
[[nodiscard]] constexpr index_t flops(const Path &chosen,
                                      const BoundExtents &bound) noexcept {
  return std::ranges::fold_left(
      chosen.steps, index_t{0}, [&](const index_t acc, const Step &s) {
        return acc + volume(s.target, bound) * volume(s.k, bound);
      });
}

} // namespace impl

} // namespace einsum
