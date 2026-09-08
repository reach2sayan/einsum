"""The object einsum() answers, and what it can be asked."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import einsum


def test_queries():
    plan = einsum.einsum("ij,jk->ik")
    assert plan.subscripts == "ij,jk->ik"
    assert plan.operand_count == 2
    assert plan.output_labels == "ik"
    assert repr(plan) == "Einsum('ij,jk->ik')"


def test_implicit_output_is_reported_as_written():
    assert einsum.einsum("ij,jk").subscripts == "ij,jk"
    # NumPy's rule: the labels occurring once, in ascending order.
    assert einsum.einsum("ij,jk").output_labels == "ik"


def test_ellipsis_round_trips():
    assert einsum.parse_subscript("...ij,...jk->...ik") == "...ij,...jk->...ik"
    assert einsum.parse_subscript(" i j , j k ") == "ij,jk"


def test_reuse_is_the_point():
    plan = einsum.einsum("ij,jk->ik")
    a, b = np.ones((3, 4)), np.ones((4, 2))
    first, second = plan(a, b), plan(a, b)
    assert_allclose(first, second)
    # A different shape is a fresh lowering, not a stale one reused.
    assert plan(np.ones((5, 4)), b).shape == (5, 2)


def test_path_is_part_of_the_object():
    assert einsum.einsum("ij,jk->ik", path=einsum.Path.SEQUENTIAL) is not None
    with pytest.raises(TypeError):
        # keyword-only; the ignore is the assertion restated for mypy, which
        # rejects this call for exactly the reason the test expects it to fail
        # at run time.  Narrow on purpose: if it ever stops being a call-arg
        # error, this line stops type-checking and says so.
        einsum.einsum("ij,jk->ik", einsum.Path.SEQUENTIAL)  # type: ignore[call-arg]
