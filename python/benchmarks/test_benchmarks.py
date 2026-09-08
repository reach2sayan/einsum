"""Every case against every implementation, one cell at a time.

The ids come out as `test_contraction[<case>-<impl>]`, which is what
`--benchmark-group-by=param:case` groups the report on: one block per
contraction, the implementations sorted inside it.
"""

from __future__ import annotations

import numpy as np
import pytest
from cases import CASES
from impls import IMPLS
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    "impl", [pytest.param(i, id=i.name, marks=i.marks) for i in IMPLS])
@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in CASES])
def test_contraction(benchmark, case, impl):
    operands = case.operands()
    run = impl.prepare(case, operands)
    if run is None:
        pytest.skip(f"no {impl.name} spelling of {case.subscripts}")

    # Checked before it is timed.  A table where a wrong answer can win is
    # worse than no table.
    #
    # An atol as well as an rtol, because the operands are standard normal and
    # a summed row can land near zero: one element of a 64-term float32 product
    # came out 1.5e-6 apart, which is nothing in absolute terms and 1e-4 in
    # relative ones.  These are two orderings of the same sum, not two answers.
    rtol, atol = (1e-4, 1e-5) if case.dtype == np.float32 else (1e-10, 1e-12)
    assert_allclose(
        np.asarray(run()),
        np.einsum(case.subscripts, *operands, optimize=True),
        rtol=rtol, atol=atol,
    )

    benchmark.extra_info["subscripts"] = case.subscripts
    benchmark.extra_info["shapes"] = str(case.shapes)
    benchmark.extra_info["dtype"] = case.dtype.name
    benchmark(run)
