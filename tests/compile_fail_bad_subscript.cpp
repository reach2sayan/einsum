// Must not compile: the subscript has a character the grammar refuses, and the
// static_assert generated from EINSUM_ERRC_SEQ says which one it was.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  std::vector A{1, 2, 3, 4};
  std::mdspan<int, std::extents<std::size_t, 2, 2>> a{A.data()};
  auto ein = einsum::einsum<"i1,jk->ik">(a, a);
  ein.eval();
  return ein.get_result()[0];
}
