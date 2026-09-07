// Must not compile: j is bound to 3 by the first operand and to 4 by the
// second, and a compile-time operand knows both extents already.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  std::vector<int> A(6);
  std::vector<int> B(20);
  std::mdspan<int, std::extents<std::size_t, 2, 3>> a{A.data()};
  std::mdspan<int, std::extents<std::size_t, 4, 5>> b{B.data()};
  auto ein = einsum::einsum<"ij,jk->ik">(a, b);
  ein.eval();
  return ein.get_result()[0];
}
