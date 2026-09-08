"""What the comparison runs against, and on how many cores.

Two things have to be true before the first number is believable, and both
have to be arranged before anything here imports NumPy.

One core.  OpenBLAS sizes its thread pool when the library loads, so
`OMP_NUM_THREADS` set after `import numpy` does nothing at all -- the row would
be a sixteen-core number wearing a single-threaded label.  A conftest runs
before the test module, which is early enough in the normal case; when it is
not, because a plugin imported NumPy first, this refuses to run rather than
report.  Single-threaded is also the honest comparison: this einsum has no
thread pool, and a BLAS that fans out is measuring the machine.

The build tree.  Same guard as python/tests/conftest.py, and it matters more
here: a stale wheel shadowing the build you just changed makes the tests fail
loudly but makes the benchmark merely wrong.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

_POOLS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _var in _POOLS:
    os.environ.setdefault(_var, "1")

if "numpy" in sys.modules:
    raise RuntimeError(
        "numpy was imported before this conftest could pin the BLAS thread "
        "pools, so every number below would be a multi-threaded one.  Re-run "
        "with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 in "
        "the environment."
    )

# The allocator, which glibc also reads only at startup.  A result big enough
# to be mmapped goes back to the kernel on free, so the next call faults in 128
# fresh zeroed pages and overwrites them -- 455us of the matmul_256 row.  It is
# a real cost, but not one every process pays: importing torch retunes glibc
# and moves that row by 2.5x all on its own.  Left unpinned, whether torch
# happens to be installed decides what every other row measures.
if any(os.environ.get(v) is None
       for v in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_")):
    warnings.warn(
        "MALLOC_MMAP_THRESHOLD_ and MALLOC_TRIM_THRESHOLD_ are unset, so rows "
        "whose result is larger than glibc's mmap threshold measure the "
        "allocator as much as the kernel, and are not comparable with a run "
        "that has them.  The python_benchmark_json target sets both; setting "
        "only one leaves glibc's dynamic tuning off for the other.",
        RuntimeWarning, stacklevel=1)

# Without pytest-benchmark the `benchmark` fixture does not exist and every
# test in here errors.  `uv sync --group bench` is the fix; until then the
# directory collects to nothing, which is a clearer thing to read.
if importlib.util.find_spec("pytest_benchmark") is None:
    collect_ignore_glob = ["test_*.py"]

import einsum  # noqa: E402  -- after the thread pinning above, on purpose

_EXPECTED = os.environ.get("EINSUM_PACKAGE_DIR")
if _EXPECTED and Path(einsum.__file__).parent != Path(_EXPECTED).resolve() / "einsum":
    raise RuntimeError(
        f"imported einsum from {Path(einsum.__file__).parent}, not the build tree "
        f"at {_EXPECTED}.  An editable install shadows it; `pip uninstall einsum` "
        f"or install non-editable."
    )
