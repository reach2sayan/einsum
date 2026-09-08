"""Einstein summation over NumPy arrays, with the subscript parsed at run time.

    >>> import numpy as np, einsum
    >>> a, b = np.ones((3, 4)), np.ones((4, 2))
    >>> einsum.contract("ij,jk->ik", a, b).shape
    (3, 2)

``contract`` is the one-shot spelling. Keep the object instead when the same
subscript runs more than once -- it caches the lowering it produced, keyed on
the operands' shapes *and* strides, so a repeated call of the same shape skips
inferring and lowering again:

    >>> matmul = einsum.einsum("ij,jk->ik")
    >>> matmul(a, b).shape
    (3, 2)

Operands are float64 or float32 -- float32 throughout is a float32 call, and
anything else is converted to float64, which is what NumPy's own promotion
would have done. An array is read where it lies whatever its strides: C- or
F-ordered, sliced, transposed, reversed. Only a broadcast view is materialised.

The result comes back at the rank the *subscript* implies, so ``"i,j->ij"`` and
``"ij,kl->ijkl"`` are arrays here and not errors.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ._einsum import (
    MAX_OPERANDS,
    MAX_RANK,
    Einsum,
    Path,
    __version__,
    einsum,
    errc,
    parse_subscript,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    import einsum._einsum as _extension

    # What the exception translator adds at run time and a generated stub
    # cannot see: the code the refusal carries.
    class Error(_extension.Error):
        """A refusal from einsum, carrying the code it refused with."""

        code: errc
else:
    from ._einsum import Error

__all__ = [
    "MAX_OPERANDS",
    "MAX_RANK",
    "Einsum",
    "Error",
    "Path",
    "__version__",
    "contract",
    "einsum",
    "errc",
    "parse_subscript",
]


# The parse is the expensive half of a one-shot call, and a subscript is a
# small, repeated string.  Sharing the object is safe because __call__ holds
# the GIL for its whole length: the caches it writes are never written twice at
# once.
@lru_cache(maxsize=256)
def _plan(subscripts: str, path: Path) -> Einsum:
    return einsum(subscripts, path=path)


def contract(subscripts: str, /, *operands: ArrayLike,
             path: Path = Path.GREEDY) -> NDArray[np.floating]:
    """Contract the operands over `subscripts` -- ``np.einsum``'s spelling.

    Equivalent to ``einsum(subscripts, path=path)(*operands)``, with the parsed
    subscript remembered between calls.
    """
    return _plan(subscripts, path)(*operands)
