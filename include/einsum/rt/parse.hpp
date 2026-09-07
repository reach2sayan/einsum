#pragma once

#include "einsum/core/plan.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/export.hpp"

#include <string_view>

// A subscript that is not known until run time.  The grammar behind these is
// the only part of the library that is compiled rather than included, because
// Boost.Parser's parse() is a try/catch and everything else here is built
// -fno-exceptions.  Link einsum::rt.
namespace einsum::rt {

// The subscript alone, for cross-checking against impl::parse_subscripts.
[[nodiscard]] EINSUM_API result<Subscripts> parse(std::string_view source) noexcept;

// The whole pipeline: parse, then lower.  What a caller wants.
[[nodiscard]] EINSUM_API result<Plan> plan(std::string_view source) noexcept;

} // namespace einsum::rt
