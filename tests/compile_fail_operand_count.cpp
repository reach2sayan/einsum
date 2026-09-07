// Must not compile: the subscript names three operands and the call passes two.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  std::vector A{1, 2, 3, 4};
  std::mdspan<int, std::extents<std::size_t, 2, 2>> a{A.data()};
  auto ein = einsum::einsum<"ij,jk,kl->il">(a, a);
  ein.eval();
  return ein.get_result()[0];
}
