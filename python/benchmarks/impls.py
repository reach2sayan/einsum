"""Who is being compared, and what each one is given before the clock starts.

Every adapter takes a case and its operands and hands back a *thunk*: a
zero-argument callable with all the setup already done. That is where the
whole fairness question lives, because each of these has a cost that is not
the contraction --

  einsum          parses the subscript, then lowers it against the operands'
                  shapes and strides and caches the result; one warm-up call
                  pays both.
  np.einsum       with optimize=True re-derives the contraction path on every
                  single call.  That row is kept, because it is what a caller
                  who typed `optimize=True` actually pays -- but it gets a
                  sibling row handed the path np.einsum_path already found,
                  which is the like-for-like comparison.
  opt_einsum      contract_expression is its precompiled form.
  torch           from_numpy is a view over the same buffer, so no copy is
                  inside the timer.

Timing a cold call for one and a warm call for another would make the table a
statement about caches rather than about kernels.
"""

from __future__ import annotations

import contextlib
import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import einsum

if TYPE_CHECKING:
    from collections.abc import Callable

    from cases import Array, Case

    Thunk = Callable[[], Any]
    Prepare = Callable[[Case, "tuple[Array, ...]"], "Thunk | None"]

# torch is opt-in: a ~200MB wheel for one row.  A skipif mark rather than an
# importorskip at module scope, so its absence shows up as a visible `s` in the
# report instead of the row quietly not existing.
_HAS_TORCH = importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class Impl:
    """One competitor, and the marks that decide whether it can run here."""

    name: str
    prepare: Prepare
    marks: tuple[pytest.MarkDecorator, ...] = field(default=())

    def __str__(self) -> str:
        """The implementation's name, which is also its id in the report."""
        return self.name


def _einsum(path: einsum.Path) -> Prepare:
    def prepare(case: Case, operands: tuple[Array, ...]) -> Thunk:
        plan = einsum.einsum(case.subscripts, path=path)
        plan(*operands)  # parse and lower once; the cache is keyed on layout
        return lambda: plan(*operands)

    return prepare


def _numpy(case: Case, operands: tuple[Array, ...]) -> Thunk:
    return lambda: np.einsum(case.subscripts, *operands, optimize=False)


def _numpy_optimize(case: Case, operands: tuple[Array, ...]) -> Thunk:
    # The path search stays inside the loop.  That is the point of this row.
    return lambda: np.einsum(case.subscripts, *operands, optimize=True)


def _numpy_path(case: Case, operands: tuple[Array, ...]) -> Thunk:
    path, _ = np.einsum_path(case.subscripts, *operands, optimize="optimal")
    return lambda: np.einsum(case.subscripts, *operands, optimize=path)


def _opt_einsum(case: Case, operands: tuple[Array, ...]) -> Thunk:
    import opt_einsum as oe

    expr = oe.contract_expression(case.subscripts, *(o.shape for o in operands))
    return lambda: expr(*operands)


def _torch(case: Case, operands: tuple[Array, ...]) -> Thunk:
    import torch

    torch.set_num_threads(1)
    # Refused once the pool has started, which it will have been by the second
    # case; the first call is the one that counts.
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    tensors = [torch.from_numpy(o) for o in operands]  # a view, not a copy
    return lambda: torch.einsum(case.subscripts, *tensors)


def _native(case: Case, operands: tuple[Array, ...]) -> Thunk | None:
    """The hand-written NumPy floor, or None where the case has none."""
    if case.native is None:
        return None
    native = case.native
    return lambda: native(*operands)


IMPLS: list[Impl] = [
    Impl("einsum_greedy", _einsum(einsum.Path.GREEDY)),
    Impl("einsum_sequential", _einsum(einsum.Path.SEQUENTIAL)),
    Impl("np_einsum", _numpy),
    Impl("np_einsum_optimize", _numpy_optimize),
    Impl("np_einsum_path", _numpy_path),
    Impl("opt_einsum", _opt_einsum),
    Impl("torch_einsum", _torch, (pytest.mark.skipif(
        not _HAS_TORCH,
        reason="torch is opt-in: uv pip install torch --torch-backend=cpu"),)),
    Impl("native_numpy", _native),
]
