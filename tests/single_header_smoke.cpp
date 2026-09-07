// The amalgamated header on its own: no include path into include/, nothing
// from src/.  Compiled with -fsyntax-only by CI, so it proves the single header
// is self-contained -- which is the only thing about it that can rot.
#include "einsum.hpp"

#include <vector>

int main() {
  std::vector A{0, 1, 2, 3, 4, 5};
  std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::mdspan<int, std::extents<std::size_t, 2, 3>> a{A.data()};
  std::mdspan<int, std::extents<std::size_t, 3, 4>> b{B.data()};

  auto ein = einsum::einsum<"ij,jk->ik">(a, b);
  ein.eval();

  // The runtime lowering is header-only too; only rt::plan() needs the library.
  const auto plan = einsum::impl::build_plan("ij,jk->ik");
  std::vector<int> out(8);
  const auto ok = plan->eval(a, b, einsum::into(out, {2, 4}));
  return ein.get_result()[0] + (ok.has_value() ? 0 : 1);
}
