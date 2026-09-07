// The amalgamated header on its own: no include path into include/, nothing
// from src/.  Compiled with -fsyntax-only by CI, so it proves the single header
// is self-contained -- which is the only thing about it that can rot.
#include "einsum.hpp"

#include <vector>

int main() {
  const std::vector<std::vector<int>> a{{0, 1, 2}, {3, 4, 5}};
  const std::vector<std::vector<int>> b{
      {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

  const auto ct = einsum::einsum<"ij,jk->ik">()(a, b);

  // einsum(std::string_view) is the only part that needs the library linked, so
  // the header on its own is exercised through the compile-time form.
  const auto other = einsum::einsum<"ij,jk">()(a, b);
  return (*ct)[0][0] + (other.has_value() ? 0 : 1);
}
