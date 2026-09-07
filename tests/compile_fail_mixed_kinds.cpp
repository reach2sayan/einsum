// Must not compile: one operand is an Eigen matrix and the other a nest.  Every
// operand of one call is in the same family, and a mismatch is decided here
// rather than at run time.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  Eigen::Matrix<double, 2, 3> a;
  a.setOnes();
  const std::vector<std::vector<double>> b{{1, 2}, {3, 4}, {5, 6}};
  const auto out = einsum::einsum<"ij,jk->ik">()(a, b);
  return static_cast<int>((*out)(0, 0));
}
