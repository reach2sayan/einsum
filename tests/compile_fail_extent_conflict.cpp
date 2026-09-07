// Must not compile: the subscript repeats a label in the output, which no
// operand shape can rescue.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  const std::vector<std::vector<int>> a{{1, 2}, {3, 4}};
  const auto out = einsum::einsum<"ij->ii">()(a);
  return (*out)[0][0];
}
