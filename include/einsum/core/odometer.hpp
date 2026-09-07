#pragma once

#include "einsum/core/limits.hpp"
#include "einsum/core/view.hpp"
#include "einsum/util/ranges.hpp"

namespace einsum::impl {

// A multi-index and the memory offset it names, advanced together.  Row-major
// order -- the last axis moves fastest -- which is the order every tensor this
// library makes is laid out in, so a walk over one is a walk over the other.
//
// Incremental rather than a divide per step: pack(), scatter(), reduce(), the
// permuting copy and the GEMM's batch loop all iterate this way, and the offset
// each of them wants is a running sum.  It is the only multi-index walk here.
class Odometer {
public:
  constexpr Odometer(const Shape &ext, const Shape &str) noexcept : ext_{ext}, str_{str} {
    at_.resize(ext_.size());
  }

  constexpr explicit Odometer(const Layout &layout) noexcept
      : Odometer(layout.shape, layout.strides) {}

  [[nodiscard]] constexpr index_t offset() const noexcept { return off_; }
  [[nodiscard]] constexpr index_t count() const noexcept { return product(ext_); }

  // Carry from the last axis.  Each axis that wraps gives back exactly what it
  // has added since it last did, so no offset is ever recomputed from scratch.
  constexpr Odometer &operator++() noexcept {
    for (std::size_t i = ext_.size(); i-- > 0;) {
      off_ += str_[i];
      if (++at_[i] < ext_[i]) {
        return *this;
      }
      off_ -= str_[i] * ext_[i];
      at_[i] = 0;
    }
    return *this;
  }

private:
  Shape ext_;
  Shape str_;
  Shape at_{};
  index_t off_ = 0;
};

} // namespace einsum::impl
