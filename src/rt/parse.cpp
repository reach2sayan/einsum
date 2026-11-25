#include "einsum/rt/parse.hpp"

#include "einsum/core/einsum_object.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"

#include <string_view>

// The run-time entry point is the compile-time one called with a string that
// arrives late: the same constexpr parse_subscripts and make_plan einsum<"...">
// runs in a constant expression, run here on a std::string_view.
namespace einsum {

result<Subscripts> parse_subscript(const std::string_view source) noexcept {
  return impl::parse_subscripts(source);
}

result<Einsum> einsum(const std::string_view source, const path order) {
  return impl::parse_subscripts(source)
      .and_then(impl::make_plan)
      .transform([order](Plan &&plan) {
        return impl::einsum_access::make<impl::HeapScratch>(std::move(plan),
                                                            order);
      });
}

} // namespace einsum
