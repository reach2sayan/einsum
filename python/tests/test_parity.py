"""What the module is for: agreeing with numpy.einsum."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import einsum

# One table, exercised in both dtypes and in several memory layouts.  Each entry
# is a subscript and the shapes to run it over.
SUBSCRIPTS = [
    ("ij,jk->ik", [(3, 4), (4, 2)]),
    ("ij,jk", [(3, 4), (4, 2)]),                  # implicit output
    ("ij->ji", [(3, 4)]),
    ("ii->i", [(4, 4)]),                          # diagonal
    ("ii->", [(4, 4)]),                           # trace
    ("ij->", [(3, 4)]),
    ("ij->i", [(3, 4)]),
    ("ij,j->i", [(3, 4), (4,)]),
    ("i,i->", [(5,), (5,)]),
    ("i,j->ij", [(3,), (2,)]),                    # deeper than any operand
    ("ij,kl->ijkl", [(2, 3), (2, 2)]),            # deeper still
    ("bij,bjk->bik", [(5, 2, 3), (5, 3, 4)]),     # batched
    ("...ij,...jk->...ik", [(5, 2, 3), (5, 3, 4)]),
    ("...j,j->...", [(5, 2, 3), (3,)]),
    ("ij,jk,kl->il", [(2, 8), (8, 3), (3, 4)]),   # a path to choose
    ("ij,j->ij", [(3, 1), (4,)]),                 # size-1 broadcasting
]


def sample(shape, dtype, seed):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape) % 7 - 3 + seed


@pytest.mark.parametrize(("subscripts", "shapes"), SUBSCRIPTS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_matches_numpy(subscripts, shapes, dtype):
    operands = [sample(s, dtype, i) for i, s in enumerate(shapes)]
    got = einsum.contract(subscripts, *operands)
    want = np.einsum(subscripts, *operands)
    assert got.dtype == np.dtype(dtype)
    assert got.shape == want.shape
    assert_allclose(got, want, rtol=1e-5 if dtype is np.float32 else 1e-12)


@pytest.mark.parametrize("path", list(einsum.Path))
def test_path_does_not_change_the_answer(path):
    operands = [sample(s, np.float64, i)
                for i, s in enumerate([(2, 8), (8, 3), (3, 4)])]
    plan = einsum.einsum("ij,jk,kl->il", path=path)
    assert_allclose(plan(*operands), np.einsum("ij,jk,kl->il", *operands))


# An impl::Layout is extents and *signed* strides, so an array is read where it
# lies whatever those are.  These are the layouts a caller actually arrives in.
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(lambda x: x, id="c_order"),
        pytest.param(np.asfortranarray, id="f_order"),
        pytest.param(lambda x: x[::-1], id="reversed"),
        pytest.param(lambda x: x[:, ::2], id="sliced"),
        pytest.param(lambda x: x.T.copy().T, id="transposed"),
        pytest.param(lambda x: np.broadcast_to(x[:1], x.shape), id="broadcast"),
    ],
)
def test_strides(layout):
    base = sample((4, 6), np.float64, 1)
    a = layout(base)
    b = sample((a.shape[1], 3), np.float64, 2)
    assert_allclose(einsum.contract("ij,jk->ik", a, b), np.einsum("ij,jk->ik", a, b))


def test_operands_are_not_copied_when_they_need_not_be():
    a = sample((3, 4), np.float64, 1)
    before = a.copy()
    einsum.contract("ij->ji", a)
    assert_allclose(a, before)


def test_integer_operands_are_promoted():
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    got = einsum.contract("ij->i", a)
    assert got.dtype == np.float64
    assert_allclose(got, np.einsum("ij->i", a))


def test_mixed_precision_promotes_to_float64():
    a = sample((3, 4), np.float32, 1)
    b = sample((4, 2), np.float64, 2)
    assert einsum.contract("ij,jk->ik", a, b).dtype == np.float64


# The corners: an empty axis, a rank-0 result, and both limits at once.
def test_zero_extent():
    a, b = np.ones((0, 4)), np.ones((4, 2))
    assert einsum.contract("ij,jk->ik", a, b).shape == (0, 2)


def test_rank_zero_result():
    a = np.eye(4)
    assert_allclose(einsum.contract("ii->", a), np.einsum("ii->", a))
    assert einsum.contract("ii->", a).shape == ()


def test_the_operand_limit_is_reachable_and_enforced():
    subscript = ",".join(f"{chr(97 + i)}{chr(98 + i)}"
                         for i in range(einsum.MAX_OPERANDS))
    operands = [np.ones((2, 2))] * einsum.MAX_OPERANDS
    assert_allclose(einsum.contract(f"{subscript}->ai", *operands),
                    np.einsum(f"{subscript}->ai", *operands))

    with pytest.raises(einsum.Error) as raised:
        einsum.contract("ab,bc,cd,de,ef,fg,gh,hi,ij->aj", *([np.ones((2, 2))] * 9))
    assert raised.value.code is einsum.errc.too_many_operands


def test_the_rank_limit_is_reachable():
    x = np.ones((1,) * einsum.MAX_RANK)
    assert einsum.contract("abcdefgh->hgfedcba", x).shape == (1,) * einsum.MAX_RANK
