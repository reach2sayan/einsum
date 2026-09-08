"""The thread team: sized as asked, and invisible in the answer but for rounding.

Not to the bit: Eigen picks its block sizes from the thread count, so a
threaded product accumulates each element's sum in a different order and the
two answers differ by a few ulp -- 1e-14 on sums of order 10.  That is the
whole of the difference, and it is asserted here at a tolerance that would
catch a wrong block rather than a reordered one.  Without OpenMP in the build
the setters are honest no-ops and the count is always one.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import einsum

needs_openmp = pytest.mark.skipif(not einsum.OPENMP, reason="built without OpenMP")


@pytest.fixture
def one_thread():
    before = einsum.get_num_threads()
    einsum.set_num_threads(1)
    yield
    einsum.set_num_threads(before)


@needs_openmp
def test_the_count_round_trips(one_thread):
    einsum.set_num_threads(3)
    assert einsum.get_num_threads() == 3


def test_without_openmp_it_is_one():
    if einsum.OPENMP:
        pytest.skip("built with OpenMP")
    einsum.set_num_threads(8)
    assert einsum.get_num_threads() == 1


@needs_openmp
@pytest.mark.parametrize(("subscripts", "shapes"), [
    ("ij,jk->ik", [(256, 256), (256, 256)]),
    ("ij,jk,kl->il", [(128, 64), (64, 256), (256, 128)]),
    ("bij,bjk->bik", [(4, 128, 128), (4, 128, 128)]),
])
def test_threads_do_not_change_the_answer(one_thread, subscripts, shapes):
    rng = np.random.default_rng(0)
    operands = [rng.standard_normal(s) for s in shapes]
    plan = einsum.einsum(subscripts)
    alone = plan(*operands)
    einsum.set_num_threads(4)
    assert_allclose(plan(*operands), alone, rtol=1e-12, atol=1e-12)


@needs_openmp
def test_zero_means_the_machine(one_thread):
    # Eigen's own convention: 0 hands the choice back to OpenMP.
    einsum.set_num_threads(0)
    assert einsum.get_num_threads() >= 1
