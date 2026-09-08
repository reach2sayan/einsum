"""Every refusal, and the code it carries."""

from __future__ import annotations

import numpy as np
import pytest

import einsum


@pytest.mark.parametrize(
    ("subscripts", "code"),
    [
        ("ij,jk->i k!", einsum.errc.bad_syntax),
        ("i...j...,jk", einsum.errc.ellipsis_repeated),
        ("ij,,jk", einsum.errc.empty_operand),
        ("abcdefghi", einsum.errc.rank_too_high),
        ("ij,jk->iz", einsum.errc.unknown_output_label),
        ("ij,jk->ii", einsum.errc.repeated_output_label),
    ],
)
def test_the_parse_refuses(subscripts, code):
    with pytest.raises(einsum.Error) as raised:
        einsum.einsum(subscripts)
    assert raised.value.code is code


@pytest.mark.parametrize(
    ("subscripts", "shapes", "code"),
    [
        ("ij,jk->ik", [(3, 4)], einsum.errc.operand_count_mismatch),
        ("ij,jk->ik", [(3, 4), (4, 2), (2, 2)], einsum.errc.operand_count_mismatch),
        ("ij,jk->ik", [(3, 4), (5, 2)], einsum.errc.extent_conflict),
        ("ij,jk->ik", [(3, 4, 2), (4, 2)], einsum.errc.rank_mismatch),
        ("...i,...i->...i", [(2, 3), (5, 3)], einsum.errc.broadcast_mismatch),
    ],
)
def test_the_call_refuses(subscripts, shapes, code):
    operands = [np.ones(s) for s in shapes]
    with pytest.raises(einsum.Error) as raised:
        einsum.einsum(subscripts)(*operands)
    assert raised.value.code is code


def test_the_error_reads_as_a_sentence():
    with pytest.raises(einsum.Error, match="one label is bound to two different"):
        einsum.contract("ij,jk->ik", np.ones((3, 4)), np.ones((5, 2)))


def test_a_rank_past_the_limit_is_refused():
    too_deep = np.ones((1,) * (einsum.MAX_RANK + 1))
    with pytest.raises(einsum.Error) as raised:
        einsum.contract("...->...", too_deep)
    assert raised.value.code is einsum.errc.rank_too_high


def test_something_that_is_not_an_array():
    with pytest.raises(TypeError):
        # Passing something that is not an array is the whole test, so mypy
        # objecting is the annotation working rather than a defect to fix.
        einsum.contract("ij->ji", object())  # type: ignore[arg-type]
