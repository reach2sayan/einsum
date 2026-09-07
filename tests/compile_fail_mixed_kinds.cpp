// Must not compile: one operand is an Eigen matrix and the other an mdspan.
// Every operand of one einsum stands on the same rung of the ladder in
// core/kind.hpp, and a mismatch is decided here rather than at run time.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  std::vector<double> B(6);
  Eigen::Matrix<double, 2, 3> a;
  a.setOnes();
  std::mdspan<double, std::extents<std::size_t, 3, 2>> b{B.data()};
  auto ein = einsum::einsum<"ij,jk->ik">(a, b);
  ein.eval();
  return static_cast<int>(ein.get_result()[0]);
}
