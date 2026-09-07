# EinsteinSummation

A header-only einsum for C++23: one object, one call, and the result comes back
by value in the operands' own family.

```cpp
#include <einsum/einsum.hpp>

// The subscript is a template argument, so a bad one is a static_assert
// carrying the sentence -- not a template backtrace.
const auto matmul = einsum::einsum<"ij,jk->ik">();
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
| Compiler | GCC 14+ or Clang 20+ |
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
`g++` on many distributions is still 13.

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

Three, because everything else is one variable on top of one of them rather than
a preset of its own:

```
CXX=clang++-20 cmake --preset release          # the Clang 20 build
cmake --preset release -DEINSUM_TRACE=ON       # -ftime-trace / -ftime-report
cmake --preset asan -DEINSUM_SANITIZE=thread   # or undefined
```

Options: `EINSUM_BUILD_TESTS`, `EINSUM_BUILD_BENCHMARKS`, `EINSUM_BUILD_SHARED_RT`,
`EINSUM_NATIVE_ARCH`, `EINSUM_SANITIZE` (`off`/`address`/`undefined`/`thread`),
`EINSUM_TRACE`.

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
value at all, because no operand type implies that rank. On the compile-time
path, name the result instead and its own type sets the rank:

```cpp
std::vector<std::vector<std::vector<std::vector<int>>>> out;
einsum<"ij,kl->ijkl">()(a, b, out);          // the output form fixes the rank
```

On the runtime path that form does not exist (see below), so a deeper output is
`rank_mismatch` and nothing more — use the compile-time spelling for those
subscripts.

The compile-time form could in principle return exact types for the cases it
*can* express, since it has the subscript as a template argument; it
deliberately returns the same ones as the runtime path, so that moving a call
between the two does not change what it answers.

## The API

There is one object and one call operator.

```cpp
const auto e = einsum::einsum<"ij,jk->ik">();   // compile time: checked here
const auto p = einsum::einsum("ij,jk->ik");     // run time: result<Einsum>

const auto c = e(a, b);                         // result<R>, by value
out = *e(a, b);                                 // "in place" is just a move
```

- **`operator()(ops...)`** converts the operands, infers the output shape,
  lowers, allocates the result, executes, and returns it. It is `const`, the
  object is immutable, and there is no caching of anything observable.
- **`operator()(ops..., out)`** exists only on the compile-time form, where the
  operand count is a constant so one extra argument is unambiguously an output.
  The runtime form has no such luxury — a trailing non-const matrix could not be
  told from one more operand — so use the move above.
- **`subscripts()`**, **`operand_count()`**, **`output_labels()`** are the only
  queries.

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

## The single header

```
python3 scripts/amalgamate.py     # -> single_include/einsum.hpp
```

Covers the header-only library: `einsum<"...">`, the object it answers, the views, the
lowering and the kernels. It still needs Boost, Eigen and mdspan on the include
path. It does **not** cover `einsum(std::string_view)` — that grammar is compiled into
`libeinsum_rt`, and a single header cannot carry a translation unit.

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

## Tracing and sanitizers

```
cmake --preset release -DEINSUM_TRACE=ON && cmake --build --preset release
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
```

`EINSUM_TRACE=ON` turns ccache off, because a cache hit restores the object and
never writes the `-ftime-trace` JSON beside it.

## Licence

See `LICENSE.txt`.
