// Both entry points, side by side: the subscript known when the program was
// compiled, and the one that was not.
#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <print>
#include <vector>

int main() {
  // --- compile time --------------------------------------------------------
  // The subscript is a template argument, so the result's extents, its size and
  // the scratch it needs are all in the type; a bad subscript here is a
  // static_assert carrying the sentence, not a template backtrace.
  const std::vector A{1, 2, 3, 4, 5, 6};
  const std::vector B{7, 8, 9, 10, 11, 12};
  const std::mdspan<const int, std::extents<std::size_t, 2, 3>> a{A.data()};
  const std::mdspan<const int, std::extents<std::size_t, 3, 2>> b{B.data()};

  auto product = einsum::einsum<"ij,jk->ik">(a, b);
  product.eval();
  std::println("einsum<\"ij,jk->ik\">  ->  {}x{}", product.output_shape[0],
               product.output_shape[1]);
  for (std::size_t i = 0; i < 2; ++i) {
    std::println("  {} {}", (product.result()[i, 0]), (product.result()[i, 1]));
  }

  auto frobenius = einsum::einsum<"ij,ij->">(a, a);
  frobenius.eval();
  std::println("einsum<\"ij,ij->\">    ->  {}", frobenius.get_result()[0]);

  // --- run time ------------------------------------------------------------
  // The same lowering, from a string nobody knew until now.  Every failure is
  // an expected, so a subscript that does not parse is a value to print.
  const auto plan = einsum::rt::plan("ij,jk->ik");
  if (!plan) {
    std::println("rt::plan failed: {}", plan.error());
    return 1;
  }

  const Eigen::MatrixXd left = Eigen::MatrixXd::Identity(3, 3) * 2.0;
  const Eigen::MatrixXd right = Eigen::MatrixXd::Ones(3, 2);
  Eigen::MatrixXd out(3, 2);

  // Eigen operands in, Eigen result out -- written straight into `out`, with
  // the workspace the only allocation and none at all after bind().
  if (const auto ok = (*plan)(left, right, einsum::into(out)); !ok) {
    std::println("eval failed: {}", ok.error());
    return 1;
  }
  std::println("rt::plan(\"ij,jk->ik\")  ->  {}x{}, first row {} {}", out.rows(),
               out.cols(), out(0, 0), out(0, 1));

  // And the same call without naming an output: the result comes back in the
  // kind the operands were.
  const auto owned = (*plan)(left, right);
  std::println("owning form           ->  {}x{}", owned->rows(), owned->cols());

  const auto bad = einsum::rt::plan("ij,,jk");
  std::println("rt::plan(\"ij,,jk\")    ->  {}", bad.error());
  return 0;
}
