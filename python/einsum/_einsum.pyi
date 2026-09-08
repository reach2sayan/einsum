"""
einsum's runtime path: a subscript parsed at run time, and the object it answers.  Import einsum instead.
"""
from __future__ import annotations
import enum
import typing
__all__: list[str] = ['Einsum', 'Error', 'MAX_OPERANDS', 'MAX_RANK', 'OPENMP', 'Path', 'einsum', 'errc', 'get_num_threads', 'parse_subscript', 'set_num_threads']
class Einsum:
    """
    A parsed subscript.  Call it with arrays.
    """
    def __call__(self, *args) -> typing.Any:
        """
        Contract the operands and hand back the result as an ndarray, at the rank the subscript implies.
        """
    def __repr__(self) -> str:
        ...
    @property
    def operand_count(self) -> int:
        """
        How many operands the subscript names.
        """
    @property
    def output_labels(self) -> str:
        """
        The output's labels, in the order the result is indexed.
        """
    @property
    def subscripts(self) -> str:
        """
        The subscript this was built from.
        """
class Error(ValueError):
    pass
class Path(enum.IntEnum):
    """
    In what order the operands are contracted.
    """
    GREEDY: typing.ClassVar[Path]  # value = <Path.GREEDY: 0>
    SEQUENTIAL: typing.ClassVar[Path]  # value = <Path.SEQUENTIAL: 1>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class errc(enum.IntEnum):
    """
    Why einsum refused; Error.code carries one.
    """
    bad_syntax: typing.ClassVar[errc]  # value = <errc.bad_syntax: 0>
    broadcast_mismatch: typing.ClassVar[errc]  # value = <errc.broadcast_mismatch: 11>
    ellipsis_not_in_output: typing.ClassVar[errc]  # value = <errc.ellipsis_not_in_output: 12>
    ellipsis_repeated: typing.ClassVar[errc]  # value = <errc.ellipsis_repeated: 1>
    empty_operand: typing.ClassVar[errc]  # value = <errc.empty_operand: 2>
    extent_conflict: typing.ClassVar[errc]  # value = <errc.extent_conflict: 10>
    no_operands: typing.ClassVar[errc]  # value = <errc.no_operands: 3>
    operand_count_mismatch: typing.ClassVar[errc]  # value = <errc.operand_count_mismatch: 8>
    output_mismatch: typing.ClassVar[errc]  # value = <errc.output_mismatch: 13>
    rank_mismatch: typing.ClassVar[errc]  # value = <errc.rank_mismatch: 9>
    rank_too_high: typing.ClassVar[errc]  # value = <errc.rank_too_high: 5>
    repeated_output_label: typing.ClassVar[errc]  # value = <errc.repeated_output_label: 7>
    too_many_operands: typing.ClassVar[errc]  # value = <errc.too_many_operands: 4>
    unknown_output_label: typing.ClassVar[errc]  # value = <errc.unknown_output_label: 6>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
def einsum(subscripts: str, *, path: Path = ...) -> Einsum:
    """
    Parse a subscript and hand back the object that runs it.
    """
def get_num_threads() -> int:
    """
    How many threads a matrix product will use.  1 in a build without OpenMP.
    """
def parse_subscript(subscripts: str) -> str:
    """
    The subscript, parsed and written back out.  Raises Error with the code the grammar refused with.
    """
def set_num_threads(n: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    How many threads a matrix product may use; 0 hands the choice back to OpenMP.  A no-op in a build without OpenMP.
    """
MAX_OPERANDS: int = 8
MAX_RANK: int = 8
OPENMP: bool = True
__version__: str = '2.0.0'
