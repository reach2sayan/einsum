#pragma once

#include "einsum/core/einsum_object.hpp"
#include "einsum/core/path.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/export.hpp"

#include <string_view>

// A subscript that is not known until run time.  Parsed and planned by the same
// constexpr functions einsum<"..."> evaluates at compile time; only these two
// entry points are compiled, into einsum::rt.
namespace einsum {

// The subscript alone, as impl::parse_subscripts reads it.
[[nodiscard]] EINSUM_API result<Subscripts>
parse_subscript(std::string_view source) noexcept;

// Parse, lower, and hand back the object a caller calls.  The runtime half of
// the entry point; the compile-time half is the einsum<"..."> overload in
// ct/einsum.hpp.
[[nodiscard]] EINSUM_API result<Einsum> einsum(std::string_view source,
                                               path order = path::greedy);

} // namespace einsum
