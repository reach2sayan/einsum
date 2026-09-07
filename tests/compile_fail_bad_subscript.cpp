// Must not compile: the subscript has a character the grammar refuses, and the
// static_assert generated from EINSUM_ERRC_SEQ says which one it was.
#include "einsum/einsum.hpp"

#include <vector>

int main() {
  const std::vector<std::vector<int>> a{{1, 2}, {3, 4}};
  const auto out = einsum::einsum<"i1,jk->ik">()(a, a);
  return (*out)[0][0];
}
