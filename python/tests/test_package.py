"""The package, rather than what it computes."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import einsum
import einsum._einsum as extension


def test_all_covers_the_extension():
    exported = {n for n in dir(extension) if not n.startswith("_")}
    assert exported <= set(einsum.__all__)


def test_py_typed_ships():
    assert files("einsum").joinpath("py.typed").is_file()


def test_one_version():
    written = Path(__file__).parents[2] / "CMakeLists.txt"
    found = re.search(r"project\(EinsteinSummation VERSION ([0-9.]+)",
                      written.read_text())
    assert found is not None
    assert einsum.__version__ == found.group(1)


def test_limits_are_the_library_s():
    assert einsum.MAX_RANK == 8
    assert einsum.MAX_OPERANDS == 8
