// Both entry points, side by side: the subscript known when the program was
// compiled, and the one that was not.  One object, one call, and the result
// comes back by value in the operands' own family.
#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <print>
#include <vector>

int main() {
  // --- compile time --------------------------------------------------------
  // The subscript is a template argument, so a bad one here is a static_assert
  // carrying the sentence, not a template backtrace -- and the operand count is
  // a constant, which is what lets this form also take an output.
  const auto product = einsum::einsum<"ij,jk->ik">();

  const std::vector<std::vector<int>> a{{1, 2, 3}, {4, 5, 6}};
  const std::vector<std::vector<int>> b{{7, 8}, {9, 10}, {11, 12}};

  const auto c = product(a, b);
  std::println("einsum<\"ij,jk->ik\">  ->  {}x{}", c->size(), (*c)[0].size());
  for (const auto &row : *c) {
    std::println("  {} {}", row[0], row[1]);
  }

  const auto frobenius = einsum::einsum<"ij,ij->">()(a, a);
  std::println("einsum<\"ij,ij->\">    ->  {}", (*frobenius)[0][0]);

  // --- run time ------------------------------------------------------------
  // The same lowering from a string nobody knew until now.  Every failure is an
  // expected, so a subscript that does not parse is a value to print.
  const auto plan = einsum::einsum("ij,jk->ik");
  if (!plan) {
    std::println("einsum(\"ij,jk->ik\") failed: {}", plan.error());
    return 1;
  }

  const Eigen::MatrixXd left = Eigen::MatrixXd::Identity(3, 3) * 2.0;
  const Eigen::MatrixXd right = Eigen::MatrixXd::Ones(3, 2);

  // Eigen operands in, an Eigen matrix out, moved rather than copied.
  const auto out = (*plan)(left, right);
  if (!out) {
    std::println("call failed: {}", out.error());
    return 1;
  }
  std::println("einsum(\"ij,jk->ik\")   ->  {}x{}, first row {} {}",
               out->rows(), out->cols(), (*out)(0, 0), (*out)(0, 1));

  const auto bad = einsum::einsum("ij,,jk");
  std::println("einsum(\"ij,,jk\")     ->  {}", bad.error());
  return 0;
}
