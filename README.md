# einsum

A header-only einsum for C++23: one object, one call, and the result comes back
by value in the operands' own family.

```cpp
#include <einsum/einsum.hpp>
using einsum::literals::operator""_ct;

// The subscript reaches a template parameter, so a bad one is a static_assert
// carrying the sentence -- not a template backtrace.
const auto matmul = einsum::einsum("ij,jk->ik"_ct);
const auto c = matmul(a, b);                  // result<...>, by value
```

```cpp
#include <einsum/rt/parse.hpp>          // needs libeinsum_rt linked

// The same lowering from a subscript nobody knew until now.
const auto plan = einsum::einsum("ij,jk->ik");
const auto c = (*plan)(a, b);
```

Every failure travels through `std::expected`; nothing in any header throws.

## Requirements

| | |
|---|---|
| Compiler | GCC 14+ or Clang 20+; MSVC 2022 (17.10+) is built but not promised |
| Standard | C++23 |
| CMake | 3.26+ |

Clang 18 and 19 are below the floor: libstdc++ gates `<expected>` on
`__cpp_concepts` at a level they do not define, so the header is present and
declares nothing. `std::views::enumerate` and `std::from_range`, which the
headers use throughout, need libstdc++ 14 -- with Clang it is the standard
library that decides, so a new Clang against an old libstdc++ lands here too. A
toolchain probe compiles all three at configure time and names the reason rather
than letting the build fail a thousand lines in.

The top-level `CMakeLists.txt` picks `g++-15` or `g++-14` off `PATH` before
`project()` unless you set `CMAKE_CXX_COMPILER` or `CXX`, because the default
`g++` on many distributions is still 13. Not on Windows, where the compiler is
whatever the developer prompt put there.

### Windows

MSVC builds in CI and is not gated on: the two Windows jobs carry
`continue-on-error`, so a red Windows leg does not fail a pull request. The
build is watched, in other words, rather than promised. There is no Windows
preset -- the presets pin no compiler, so `cmake --preset release` from a
developer command prompt is the whole of it, and CI names `cl` on the command
line rather than in a preset.

Two things differ from the Linux build by design rather than by omission:

- **The vendored header list is not checked.** `scripts/vendor_headers.py` asks
  the compiler for the include closure with `-M`, which `cl.exe` has no
  spelling for, so `EINSUM_CHECK_VENDORED_HEADERS` defaults off on Windows and
  Linux stays the authority for `cmake/EinsumVendoredHeaders.cmake` -- as it is
  for `python/einsum/_einsum.pyi`. The list already ships the MSVC arms of
  `boost/config` and `boost/preprocessor`, which are vendored whole, so an
  installed einsum is consumable from MSVC regardless of which compiler derived
  it.
- **`tests_noalloc` does not run.** Replacing `operator new` is program-wide on
  ELF; a DLL keeps the CRT's. The suite is a statement about the library's
  allocation behaviour rather than about the platform, and Linux is where it can
  still be made.

## Dependencies

All three are fetched and SHA-verified; none is ever `find_package`d, so no
machine's idea of "Boost" can change what this builds against.

| | Pin | Used for |
|---|---|---|
| Boost | 1.92.0 (`b2-nodocs` tarball) | Parser, Mp11, Preprocessor, Container, Test |
| Eigen | 3.4.0 | every kernel |
| kokkos mdspan | commit `80fc772e` | `<experimental/mdspan>`; no libstdc++ on the floor ships `<mdspan>` |
| Google Benchmark | 1.9.1 | benchmarks only |

Sources are unpacked into `<repo>/.deps`, outside every build tree, so
`rm -rf build` does not re-download them.

To build against a copy you already have:

```
cmake --preset release \
  -DEINSUM_BOOST_INCLUDEDIR=/path/to/boost-1.92.0 \
  -DEINSUM_EIGEN_INCLUDEDIR=/usr/include/eigen3
```

Either variable names the directory *containing* `boost/` or `Eigen/`, and
setting it turns that fetch off entirely.

## Building

```
cmake --preset release && cmake --build --preset release && ctest --preset release
```

| Preset | |
|---|---|
| `debug` | unoptimised, assertions live |
| `release` | `-O3 -march=native` |
| `asan` | Debug + AddressSanitizer, everything instrumented including Boost.Test |
| `python` | Release + the extension module, built against `.venv` |

Four, because everything else is one variable on top of one of them rather
than a preset of its own:

```
CXX=clang++-20 cmake --preset release          # the Clang 20 build
cmake --preset release -DEINSUM_TRACE=ON       # -ftime-trace / -ftime-report
cmake --preset asan -DEINSUM_SANITIZE=thread   # or undefined
```

Options: `EINSUM_BUILD_TESTS`, `EINSUM_BUILD_BENCHMARKS`, `EINSUM_BUILD_PYTHON`,
`EINSUM_INSTALL`, `EINSUM_NATIVE_ARCH`, `EINSUM_OPENMP`, `EINSUM_SANITIZE`
(`off`/`address`/`undefined`/`thread`), `EINSUM_TRACE`.

`EINSUM_OPENMP` (off by default; on in the `python` preset and the Linux
wheel) compiles consumers with `-fopenmp`, which is all Eigen needs to run a
matrix product's blocks on a thread team. There is no other change: the same
headers, and the same answer up to rounding -- Eigen picks its block sizes
from the thread count, so a sum runs in another order and lands a few ulp
away. The team is sized by `OMP_NUM_THREADS`,
or from Python by `einsum.set_num_threads(n)`; `einsum.OPENMP` says whether a
build has it at all.

Everything is built `-fno-exceptions` except two targets that cannot be: the one
object holding the Boost.Parser grammar, and the Boost.Test executables. Nothing
in any header throws.

## Operands

An operand is anything indexable, in one of two spellings, and every operand of
one call is in the same **family** with the same scalar. Ranks may differ:
`"ij,j->i"` is a matrix and a vector.

| Family | An operand is one when | A call answers |
|---|---|---|
| Eigen | it derives from `Eigen::DenseBase` and has `data()`; its `(i, j)` is read as `[i, j]` | `Eigen::Matrix<T, Dynamic, Dynamic, order-of-first-operand>` |
| nested | it is a range of ranges down to a scalar leaf — `std::vector<std::vector<T>>`, `std::array` nests, C arrays | the operand type itself when its depth is the result's, else a `std::vector` nest of that depth |
| Eigen Tensor | it has `Scalar`, `NumDimensions`, `Layout`, `dimension(i)` and `data()` — Eigen's unsupported Tensor module, detected structurally so this library never includes its header | a Tensor of the widest operand's type, which is where a rank-3 result has to live |
| view | it answers `rank()`/`extent(r)` and `x[i, j]` — `std::mdspan` and anything shaped like one | an `mdarray` of the same scalar and rank: contiguous, owned by the caller, indexed the way the operands were |

A view cannot own a result, which is why that row answers an `mdarray` rather
than a view: the library never owns an output and never hands back a reference
into itself. On the compile-time path the extents are known, so that `mdarray`
is one over a `std::array` and the call allocates nothing at all.

An operand whose memory is a strided rectangle (Eigen, `mdspan`) is handed to
Eigen where it already lives. Anything else — a vector of vectors, a proxy — is
walked once through its accessor and packed. A nest whose rows disagree is not
a tensor and answers `extent_conflict`.

Mixing families in one call is a compile-time error carrying that sentence, not
a runtime `errc`.

## The result's rank

The return type is fixed by the operand types alone, because on the runtime path
the subscript is not a type: Eigen answers a matrix, the other two families
answer a nest as deep as the widest operand. The output shape is then fitted to
that rank — a lower one is padded with extents of 1, which changes how the
result is described and not what is in it.

| Family | rank 0 | rank 1 | rank 2 | higher than the result type's rank |
|---|---|---|---|---|
| Eigen | `1x1` | `Nx1` | `MxN` | `rank_mismatch` |
| nested / view | one element behind leading axes of 1 | padded the same way | " | `rank_mismatch` |

Note which end the padding goes on: **Eigen pads trailing** — a rank-1 result is
`Nx1`, so it is read `(i, 0)` — while **nests, mdarrays and Tensors pad
leading**, so a rank-2 result from rank-3 operands is `(1, m, n)` and is read
`(0, i, j)`. The same rank-lowering subscript is therefore indexed differently
depending on the family its operands came from.

A runtime call also caches the lowering it produced, keyed on the operands' shapes *and* strides, so a repeated call of the same shape skips inferring and lowering again. It is a cache, not state: both call operators stay `const`, a copy of the object starts cold, and concurrent calls on one object need external synchronisation.

A subscript whose output is *deeper* than any operand — `"i,j->ij"` from two
vectors, `"ij,kl->ijkl"` from two matrices — therefore cannot be returned by
value at all, because no operand type implies that rank. Name the result
instead, and its own type sets the rank:

```cpp
std::vector<std::vector<std::vector<std::vector<int>>>> out;
einsum<"ij,kl->ijkl">()(a, b, out);           // compile time: one more argument
(*einsum("ij,kl->ijkl"))(a, b, einsum::out(out));   // run time: the tag says so
```

Both paths reach it. The compile-time one can count — the operand count is in
the subscript's type — so one argument past that is unambiguously an output. The
runtime one cannot, so the output says so itself: `einsum::out(x)` is a tag, and
an untagged trailing argument is an operand however it was declared.

The compile-time form could in principle return exact types for the cases it
*can* express, since it has the subscript as a template argument; it
deliberately returns the same ones as the runtime path, so that moving a call
between the two does not change what it answers.

## The API

There is one object and one call operator.

```cpp
const auto e = einsum::einsum("ij,jk->ik"_ct);  // compile time: checked here
const auto p = einsum::einsum("ij,jk->ik");     // run time: result<Einsum>

const auto c = e(a, b);                         // result<R>, by value
out = *e(a, b);                                 // "in place" is just a move
(*p)(a, b, einsum::out(out));                   // or write into it directly
```

- **`operator()(ops...)`** converts the operands, infers the output shape,
  lowers, allocates the result, executes, and returns it. It is `const`, the
  object is immutable, and there is no caching of anything observable.
- **`operator()(ops..., out)`** writes into `out` and answers `result<void>`;
  `out`'s own type fixes the result's rank, which is the only way to reach one
  the operands do not imply. On the compile-time form the trailing argument may
  be the output itself, since the operand count is a constant there. On the
  runtime form it must be `einsum::out(x)` — the count is a value, so nothing
  else could tell an output from one more operand.
- **`einsum::out(x)`** is accepted by both forms, so a call moves between the
  two paths without being rewritten.
- **`subscripts()`**, **`operand_count()`**, **`output_labels()`** are the only
  queries.
- **`"..."_ct`** is how the compile-time path takes a string. A function
  argument is never a constant expression, so the subscript has to reach a
  template parameter, and a literal operator template is the only thing that
  puts it there while still reading as a call. `einsum<"ij,jk->ik">()` is what
  the suffix expands to and remains spellable; the contraction order goes with
  it, as `"ab,bc,cd->ad"_ct.with<path::sequential>()`.

Copies of an object are independent. Concurrent calls on *one* object need
external synchronisation, because they share its scratch cache — a `mutable`
implementation detail that is grown on demand and never shrunk, and the reason
repeated calls of the same shape allocate nothing for scratch.

### The allocation guarantee, exactly

A call allocates the result it returns, and grows its object's scratch cache the
first time a shape needs more than the cache holds. A repeated call of the same
shape allocates nothing else — checked by `tests/tests_noalloc.cpp`, which
replaces `operator new` and counts.

Eigen is a separate matter and does not appear in those counts at all: its
buffers go through its own `aligned_malloc` (`std::malloc`), and its GEMM
blocking buffers live on the stack below `EIGEN_STACK_ALLOCATION_LIMIT`
(128 KB) and call `std::malloc` above it.

## Semantics

- **Implicit output** (no `->`) is NumPy's rule: the labels occurring exactly
  once across the whole input, **in ascending character order**. `"ij,jk"` infers
  `->ik`; `"ba"` infers `->ab`; `"ii"` infers `->` and is a trace.
  *This differs from v1*, which used first-appearance order.
- **A repeated label inside one operand** is a diagonal. `"ii->i"` is the
  diagonal, `"ii->"` is the trace, and the stride along a diagonal is the sum of
  the strides of the axes it crosses.
- **A label no other operand has and the output does not want** is summed away
  before the operand enters a contraction, so the GEMM runs over the smallest
  tensor that still answers the subscript.
- **Contraction order is chosen, not written.** `"ij,jk,kl->il"` is contracted in
  whichever order costs least, by a greedy search over the pairs: for a thin
  middle that is left to right, and for a fat one it is not. Pass
  `path::sequential` to get the subscript's own order back, verbatim:
  `einsum("ij,jk,kl->il", path::sequential)` or
  `einsum<"ij,jk,kl->il", path::sequential>()`.
- **Labels** are `[a-zA-Z]`. Whitespace between them is ignored.
- **`...`** stands for the axes a term does not name, at most once per term and
  at any position within it. The axes it covers are right-aligned across the
  operands, as NumPy aligns them, so a `(3, 4)` meets a `(5, 3, 4)` on their
  trailing two. An implicit output puts them first, then the labels seen exactly
  once in ascending order.
- **Size-1 broadcasting.** An axis of extent 1 meeting an axis of extent *n*
  stretches to *n*; the operand is read at one offset for every index along it.
  Two extents that are neither equal nor 1 are an error: `broadcast_mismatch`
  for a `...` axis and `extent_conflict` for a named one. A label repeated
  inside one operand walks a diagonal and must match exactly -- diagonals do not
  broadcast.

### NumPy parity

| | |
|---|---|
| implicit output (once-labels, sorted) | yes |
| diagonals, traces, partial traces | yes |
| `...` at any position, right-aligned | yes |
| size-1 broadcasting | yes |
| outputs deeper than any operand (`"i,j->ij"`) | yes, through `out(x)` |
| contraction path | greedy by default, `path::sequential` available |
| sublist form `einsum(op, [0,1], ...)` | no |
| `dtype` / `casting` / `order` arguments | no |

### Limits

`kMaxRank = 8`, `kMaxOperands = 8`. Both are the width of a `std::array` in
every constexpr structure here, so they are what a `Plan` costs whether or not
your subscript uses them.

## Errors

Every fallible operation returns `std::expected<T, einsum::error>`. There is no
message string and no source location in an `error`: it travels the numeric
path, where an allocation would be as unwelcome as the throw it replaces. The
text lives in a static table that `operator<<` and `std::formatter` read.

| `errc` | |
|---|---|
| `bad_syntax` | the subscript has a character this grammar does not accept |
| `ellipsis_repeated` | a term has more than one `...` |
| `empty_operand` | an operand between the commas has no labels |
| `no_operands` | the subscript names no operands |
| `too_many_operands` | more operands than `kMaxOperands` |
| `rank_too_high` | an operand has more labels than `kMaxRank` |
| `unknown_output_label` | the output names a label that no operand has |
| `repeated_output_label` | the output repeats a label |
| `operand_count_mismatch` | the subscript and the call disagree on how many operands there are |
| `rank_mismatch` | an operand's rank differs from the number of labels it was given |
| `extent_conflict` | one label is bound to two different extents |
| `broadcast_mismatch` | the `...` dimensions do not broadcast |
| `ellipsis_not_in_output` | the operands' `...` covers axes the output does not name |
| `output_mismatch` | the output's rank or extents are not the ones the subscript implies |

## How it is lowered

```
FixedString NTTP ─constexpr parse_subscripts─┐         Boost.Parser (src/rt/parse.cpp)
                                             ▼                     │
                                         Subscripts ◄── same struct ┘
                                             │ make_plan (constexpr)
                                             ▼
                                            Plan            extent-free, no heap
                                             │ make_geometry(plan, layouts, out)
                                             ▼
                                          Geometry     batch/M/N/K slabs, scratch offsets
                    ┌────────────────────────┴────────────────────────┐
            StaticEinsum<S>                                        Einsum
     subscript checked at compile time            subscript parsed at run time
                    └───────────────── impl::execute ──────────────────┘
                                             │
                            Eigen Map GEMM / hadamard / reduce / pack / permute
```

`make_plan` decides *what* is contracted; `make_geometry` decides *how*, given
extents and strides. Each contraction becomes a batch of `M×K · K×N` products.

The question `make_geometry` keeps asking is whether Eigen can address a group of
axes without copying it. A group collapses to one axis when, dropping the
extent-1 ones, each stride is the next stride times the next extent. If both the
row group and the column group collapse and one of them has an inner stride of
1, the rectangle is a `Map` — row-major, or column-major for the other
orientation. Otherwise it is packed into scratch first.

That inner stride of 1 is not cosmetic: Eigen's `blas_traits` refuses direct
access to an operand whose inner stride it cannot see is 1 at compile time, and
evaluates it into a heap temporary instead. Everything about the map-or-pack
decision exists to keep that from happening.

## Python

```
pip install einsum-cpp
```

```python
import numpy as np, einsum

a, b = np.ones((3, 4)), np.ones((4, 2))
einsum.contract("ij,jk->ik", a, b)            # np.einsum's spelling
matmul = einsum.einsum("ij,jk->ik")           # or keep the object
matmul(a, b)
```

The runtime path only -- the compile-time one is a template and cannot cross.
`Einsum.subscripts`, `.operand_count`, `.output_labels`, and `Path.GREEDY` /
`Path.SEQUENTIAL` are the rest of it; every refusal is an `einsum.Error` whose
`.code` is the same `errc` the C++ returns, with the same sentence.

Operands are float64 or float32 -- float32 throughout is a float32 call and
anything else is converted, which is what NumPy's own promotion would have done.
An array is read where it lies whatever its strides, so C- or F-ordered, sliced,
transposed and reversed arrays are all zero-copy; only a broadcast view is
materialised, because a repeated element is not a strided rectangle.

The result comes back at the rank the *subscript* implies rather than the
operands', so `"i,j->ij"` and `"ij,kl->ijkl"` are arrays here and need no `out`.
That is the one thing the bindings do that the typed C++ call cannot: they
allocate the output themselves, so the lowering's own shape is the answer.

A call holds the GIL for its whole length, deliberately: it grows the object's
scratch and rewrites its lowering cache, so serialising on the GIL is what makes
sharing one object between threads safe. A copy starts cold, so a caller wanting
real parallelism copies.

```
cmake --preset python && ctest --preset python   # both suites, C++ and pytest
cmake --build --preset python --target einsum_python_stubs
```

## Installing

```
cmake --install build/release --prefix /somewhere
```

```cmake
find_package(einsum REQUIRED)
target_link_libraries(app PRIVATE einsum::einsum einsum::rt)
```

One prefix and no other package. Boost, Eigen and mdspan are installed into the
same include directory as `einsum/`, because all three are fetched against a pin
here and none is one a machine reliably has -- a distribution's Boost predates
Boost.Parser, and no libstdc++ on the floor ships `<mdspan>`. What ships is the
part of each our own headers reach, worked out by asking the compiler for the
include closure:

```
python3 scripts/vendor_headers.py            # rewrite cmake/EinsumVendoredHeaders.cmake
python3 scripts/vendor_headers.py --check    # what CI runs
```

Point `EINSUM_BOOST_INCLUDEDIR` or `EINSUM_EIGEN_INCLUDEDIR` at your own copy
and that one is neither fetched nor vendored -- you asked for it, you have it.

`einsum::rt` is a shared library: `libeinsum_rt.so`, one translation unit and
two exported symbols, holding the runtime grammar.

## Benchmarks

```
cmake --build --preset release --target einsum_benchmark
./build/release/benchmark/einsum_benchmark --benchmark_filter='_double_(8|64)$'
```

`--target benchmark_json` leaves a machine-readable run beside the binary. A
sample (`-O3 -march=native`, GCC 15, one core):

```
BM_naive_double_64                 37598 ns
BM_naive_opt_double_64             37544 ns
BM_einsum_ct_double_64             11332 ns
BM_einsum_rt_double_64             11195 ns
BM_eigen_fixed_double_64           10995 ns
BM_eigen_dyn_double_64             14126 ns
```

The lowerings that are not a product are measured beside the Eigen call a
reader would have written instead, so a regression in one shows up as a ratio
rather than as a number nobody has anything to compare against:

```
BM_einsum_reduce_double/512        24815 ns
BM_rowsum_eigen_double/512         24725 ns
BM_einsum_transpose_double_64       2465 ns
BM_transpose_eigen_double_64        1410 ns
```

### Against NumPy

The same question at the Python level: is `einsum.contract` worth importing
when `np.einsum` is already there?

```
uv sync --group bench
cmake --build --preset python --target python_benchmark_json
```

`python/benchmarks/` runs each contraction against `np.einsum` (three ways:
plain, `optimize=True`, and handed a path `np.einsum_path` already found),
`opt_einsum`, `torch.einsum`, and the plain NumPy call a caller would have
written by hand -- the floor. Every plan is warm before the clock starts and
every answer is checked against `np.einsum` before it is timed. One core:
`OMP_NUM_THREADS=1`, which the target sets, because this einsum has no thread
pool and a BLAS that fans out is measuring the machine.

torch is opt-in (`uv pip install torch --torch-backend=cpu`); without it that
row skips. `pytest-benchmark compare` reads two of the saved JSON runs.

The target also pins `MALLOC_MMAP_THRESHOLD_` and `MALLOC_TRIM_THRESHOLD_`.
That is not tuning, it is repeatability: see the note below.

Minimum over the run, in microseconds, on one core of this machine
(`-march=native`, GCC 14, float64 unless the name says otherwise):

```
case            einsum   np.einsum   np(path)   opt_einsum    torch   by hand
matmul_8          0.89        2.32       9.41         8.45     9.00      0.99
matmul_16         1.26        3.65      10.09         9.57    10.14      1.22
matmul_64        11.76       64.94      20.02        20.03    23.73     10.69
matmul_256      541.88     3133.11     557.58       570.65   656.04    548.27
matmul_64_f32     6.34       47.16      17.21        15.53    15.99      5.91
batched_8x16      3.75       23.37      13.92        23.63    14.41      3.35
chain3            5.65     1975.78      21.26        22.46   113.20     21.29
chain3_bad       10.21     4479.48      25.64        25.86   108.81     37.75
chain4            2.75    21572.02      22.24        22.45   142.18      6.70
tensordot        11.64       62.77      24.08        22.63    29.06     15.19
reduce_rows      22.25       27.11      32.83        31.10    32.01     46.51
trace             0.95        1.89       6.76         4.73     6.08      1.48
diagonal          1.01        0.98       5.36         3.43     4.27      1.34
outer            31.00       74.96     123.01        79.77    36.75    125.86
matvec           28.81       82.20      33.90        30.47    42.31     21.91
transpose_64      2.00        0.86       5.91         3.44     3.48      1.76
```

Where a contraction is a chain, the path is worth more than the kernel and this
is 3x ahead of everything else -- `chain4` against plain `np.einsum` is four
orders of magnitude, which is the whole argument for choosing a path at all,
and torch, which does not choose one, is 50x behind on the same row. Where it
is one GEMM it is level with NumPy calling BLAS, at 8 and at 256 alike; where
it is a rank-one product (`outer`) it is at the cost of writing the result,
which is where torch is too and 4x from where NumPy is.

Two rows read as losses and are not. `np.einsum` answers `"ij->ji"` and
`"ii->i"` with a *view* of its input -- `np.shares_memory` says so -- and does
no work at all; this returns an array. Against the copy a caller would have to
make anyway (`x.T.copy()`, the "by hand" column) the two are level, and the
per-call floor here is under NumPy's own: 0.8us against 1.0 for `"i->i"` on
four elements.

### With threads

The table above is one core by construction. Unpinned, NumPy's BLAS and torch
use every core they can see, and until `EINSUM_OPENMP` this library used one:
at 256 and up that was a 5x deficit no kernel could make back. With it, the
same product on a team (`OMP_NUM_THREADS`, or `einsum.set_num_threads`),
minimum in microseconds, each library given the same count:

```
                          1 thread             8 threads            16 threads
                  einsum  numpy  torch   einsum  numpy  torch   einsum  numpy  torch
matmul 256           561    568    671      109    215    234      123    127    145
matmul 512          4658   4653   5338     1032   1030   1795      942    994   1225
matmul 1024        41779  38236  44054     7456   7439  14705     8986   9129  11449
chain3 256,64,1024   790    839      -      226    371      -      532    244      -
```

Level with OpenBLAS and ahead of torch at every size and count. The one thing
to know: this is an 8-core part with 16 hardware threads, and the 16-thread
column is the hyperthreads showing -- a chain of small products spends more
on the team than the team saves, and `chain3` at 16 is 2x worse than at 8.
The default is whatever `OMP_NUM_THREADS` says, and on such a machine the
physical core count is the number to say.

### A note on the allocator

`matmul_256` is the row that taught this suite to pin `MALLOC_*`. Its result is
512KB; glibc serves a block that size by mmap and returns it to the kernel on
free, so a call that allocates its result faults in 128 fresh zeroed pages and
then immediately overwrites them. `perf` puts it at 116 extra faults and ~455us
per call -- more than the arithmetic. It is a real cost, but not one every
process pays, and the thing that decides it is not this library:

```
                                   einsum   numpy a @ b
plain interpreter                  1380 us        794 us
after `import torch`                547 us        554 us
MALLOC_* thresholds raised          561 us        590 us
```

Importing torch retunes glibc, and moves this row by 2.5x without touching a
line of einsum. Unpinned, whether torch happens to be installed would decide
what every other row measured. Pinned, the kernel is level with OpenBLAS and
the allocator is out of the comparison. Writing into an output you already own
-- `einsum::out(x)` in C++ -- avoids the whole question: 654us against 724us
for the same product through a bare Eigen `noalias()`.

`transpose_64` is the remaining loss and is printed rather than dropped. It is
a copy, not a contraction, and at 2us against NumPy's 0.9 what separates them
is per-call work, not the kernel.

## Tracing and sanitizers

```
cmake --preset release -DEINSUM_TRACE=ON && cmake --build --preset release
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
```

`EINSUM_TRACE=ON` turns ccache off, because a cache hit restores the object and
never writes the `-ftime-trace` JSON beside it.

## Licence

See `LICENSE.txt`.
