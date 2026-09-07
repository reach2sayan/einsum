// Must not compile: the operands' '...' covers an axis that the explicit output
// does not name.  The operands have static extents, so the compile-time path
// knows the ranks and the refusal is a static_assert rather than a runtime
// errc -- which is the whole point of naming the subscript in the type.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  const std::vector<double> storage(2 * 2 * 2, 1.0);
  const std::mdspan<const double, std::extents<std::size_t, 2, 2, 2>> a{storage.data()};
  const auto out = einsum::einsum<"...ij,...jk->ik">()(a, a);
  return static_cast<int>(out.has_value());
}
