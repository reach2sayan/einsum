#pragma once

#include "einsum/core/einsum_object.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/export.hpp"

#include <string_view>

// A subscript that is not known until run time.  The grammar behind this is the
// only part of the library that is compiled rather than included, because
// Boost.Parser's parse() is a try/catch and everything else here is built
// -fno-exceptions.  Link einsum::rt.
namespace einsum {

// The subscript alone, for cross-checking against impl::parse_subscripts.
[[nodiscard]] EINSUM_API result<Subscripts>
parse_subscript(std::string_view source) noexcept;

// The whole pipeline: parse, lower, and hand back the object a caller calls.
// einsum("ij,jk->ik") -- the runtime half of the entry point; the compile-time
// half is the einsum<"..."> overload in ct/einsum.hpp.
[[nodiscard]] EINSUM_API result<Einsum> einsum(std::string_view source);

} // namespace einsum
