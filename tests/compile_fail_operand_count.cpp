// Must not compile: the subscript names three operands and the call passes two.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  const std::vector<std::vector<int>> a{{1, 2}, {3, 4}};
  const auto out = einsum::einsum<"ij,jk,kl->il">()(a, a);
  return (*out)[0][0];
}
