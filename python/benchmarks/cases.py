"""What gets contracted, and what a NumPy user would have written instead.

The first families are the ones `benchmark/benchmark.cpp` measures, so a row
here and a `BM_*` line there are talking about the same contraction. The sizes
are not the C++ file's full sweep: below about 16 a Python-level number is call
overhead and nothing else, so this takes 8 and 16 to measure the binding, 64 to
match the README's C++ table, and 256 to measure the kernel.

After those come the shapes the C++ suite has no reason to carry: multi-operand
chains, where the *path* rather than the kernel is what differs between one
einsum and the next, and the lowerings that are not a product at all.

Every case names a `native` -- the plain NumPy call someone would have reached
for instead. It is the floor: a row that cannot beat its own floor is a row
that says "write the NumPy". A case with no honest floor sets it to None and
the suite skips that entry rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# The suite is float64 and float32 only, which is what the binding's kernels
# are; anything else it converts, and converting is not what is being measured.
Array = np.ndarray[Any, np.dtype[np.floating[Any]]]

F64 = np.dtype(np.float64)
F32 = np.dtype(np.float32)


@dataclass(frozen=True)
class Case:
    """One contraction: what to run, over what, and the hand-written floor."""

    name: str
    subscripts: str
    shapes: tuple[tuple[int, ...], ...]
    native: Callable[..., Array] | None = None
    dtype: np.dtype[Any] = F64

    def operands(self) -> tuple[Array, ...]:
        """Fresh C-contiguous operands, the same ones on every run.

        Contiguous on purpose: this project reads an array wherever it lies,
        the competitors do not all pay the same for that, and a layout
        difference would show up in the table as a kernel difference.
        """
        rng = np.random.default_rng(0)
        return tuple(
            np.ascontiguousarray(rng.standard_normal(s, dtype=self.dtype))
            for s in self.shapes
        )

    def __str__(self) -> str:
        """The case's name, which is also its id in the report."""
        return self.name


def _chain(*ops: Array) -> Array:
    """Left-to-right matrix product -- the floor for the chain cases."""
    out = ops[0]
    for op in ops[1:]:
        out = out @ op
    return out


_MATMUL_SIZES = (8, 16, 64, 256)

CASES: list[Case] = [
    # --- the families benchmark.cpp measures ---------------------------------
    *(
        Case(f"matmul_{n}", "ij,jk->ik", ((n, n), (n, n)), np.matmul)
        for n in _MATMUL_SIZES
    ),
    # float32 has its own kernel in the binding, so it gets its own row.
    Case("matmul_64_f32", "ij,jk->ik", ((64, 64), (64, 64)), np.matmul,
         F32),
    Case("batched_8x16", "bij,bjk->bik", ((8, 16, 16), (8, 16, 16)), np.matmul),
    Case("transpose_64", "ij->ji", ((64, 64),),
         lambda a: np.ascontiguousarray(a.T)),

    # --- a path to choose ----------------------------------------------------
    # chain3 is already associative-order-sensitive; chain3_bad is shaped so
    # that left-to-right is the *wrong* order, which is the row where
    # Path.SEQUENTIAL should lose visibly and the comparison earns its keep.
    Case("chain3", "ij,jk,kl->il", ((64, 8), (8, 96), (96, 64)), _chain),
    Case("chain3_bad", "ij,jk,kl->il", ((96, 8), (8, 96), (96, 96)), _chain),
    Case("chain4", "ij,jk,kl,lm->im",
         ((32, 8), (8, 96), (96, 8), (8, 32)), _chain),
    Case("tensordot", "abcd,cdef->abef", ((8, 8, 8, 8), (8, 8, 8, 8)),
         lambda a, b: np.tensordot(a, b, axes=2)),

    # --- the lowerings that are not a product --------------------------------
    Case("reduce_rows", "ij->i", ((512, 512),), lambda a: a.sum(axis=1)),
    Case("trace", "ii->", ((512, 512),), np.trace),
    Case("diagonal", "ii->i", ((512, 512),), lambda a: np.diagonal(a).copy()),
    Case("outer", "i,j->ij", ((512,), (512,)), np.outer),
    Case("matvec", "ij,j->i", ((512, 512), (512,)), np.matmul),
]
