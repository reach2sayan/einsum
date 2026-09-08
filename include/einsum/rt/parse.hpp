#pragma once

#include "einsum/core/einsum_object.hpp"
#include "einsum/core/path.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/export.hpp"

#include <string_view>

// A subscript that is not known until run time.  Its grammar is the only part
// of the library compiled rather than included, because Boost.Parser's parse()
// is a try/catch and everything else here is built -fno-exceptions.  Link
// einsum::rt.
namespace einsum {

// The subscript alone, for cross-checking against impl::parse_subscripts.
[[nodiscard]] EINSUM_API result<Subscripts>
parse_subscript(std::string_view source) noexcept;

// Parse, lower, and hand back the object a caller calls.  The runtime half of
// the entry point; the compile-time half is the einsum<"..."> overload in
// ct/einsum.hpp.
[[nodiscard]] EINSUM_API result<Einsum> einsum(std::string_view source,
                                               path order = path::greedy);

} // namespace einsum
