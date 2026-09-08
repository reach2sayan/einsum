"""What the suite runs against.

ctest points EINSUM_PACKAGE_DIR at the package it just built.  An editable
install registers a sys.meta_path finder, which beats every sys.path entry, so
PYTHONPATH alone does not pin the module under test -- and failing silently
means the suite passes against the wrong build.
"""

from __future__ import annotations

import os
from pathlib import Path

import einsum

_EXPECTED = os.environ.get("EINSUM_PACKAGE_DIR")
if _EXPECTED and Path(einsum.__file__).parent != Path(_EXPECTED).resolve() / "einsum":
    raise RuntimeError(
        f"imported einsum from {Path(einsum.__file__).parent}, not the build tree "
        f"at {_EXPECTED}.  An editable install shadows it; `pip uninstall einsum` "
        f"or install non-editable."
    )
