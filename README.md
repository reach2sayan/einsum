# EinsteinSummation

Einstein summation for C++23, lowered onto Eigen GEMM. Write the subscript, get
the contraction — either with the subscript in the type:

```cpp
#include <einsum/einsum.hpp>

std::mdspan<const double, std::extents<std::size_t, 2, 3>> a{...};
std::mdspan<const double, std::extents<std::size_t, 3, 4>> b{...};

auto product = einsum::einsum<"ij,jk->ik">(a, b);
product.eval();
auto result = product.result();          // std::mdspan<const double, extents<2, 4>>
```

or with a subscript nobody knew until run time:

```cpp
#include <einsum/rt/parse.hpp>          // needs libeinsum_rt linked

const auto plan = einsum::rt::plan("ij,jk->ik");
if (!plan) { std::println("{}", plan.error()); return 1; }

Eigen::MatrixXd a(2, 3), b(3, 4);
const Eigen::MatrixXd c = (*plan)(a, b).value();      // Eigen in, Eigen out
```

Both spellings run the same parser output, the same `Plan`, the same lowering
and the same kernels. The only difference is *when*.

## Requirements

| | |
|---|---|
| Compiler | GCC 14+ or Clang 20+ |
| Standard | C++23 |
| CMake | 3.26+ |

Clang 18 and 19 are below the floor: libstdc++ gates `<expected>` on
`__cpp_concepts` at a level they do not define, so the header is present and
declares nothing. `std::ranges::to` needs GCC 14. A toolchain probe checks both
at configure time and names the reason rather than letting the build fail a
thousand lines in.

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

## Operand kinds

Every operand of one call must be the same *kind*, and the result comes back as
that kind. There are three, because there are three ways memory describes
itself, and each is decided by the concepts the operand satisfies rather than by
a tag anyone has to write:

| Kind | An operand is one when | `plan(ops...)` answers |
|---|---|---|
| Eigen | it derives from `Eigen::DenseBase` and has `data()` | `Eigen::Matrix<T, Dynamic, Dynamic>` (rank ≤ 2, else `not_matrix`) |
| mdspan | it has `extents_type`, `mapping_type`, `accessor_type` | an owning buffer; `.as_mdspan<R>()` names the rank |
| flat | anything else with contiguous, sized memory — `flat(v, {…})` or `flat(p, {…})` | `std::vector<T>`, row-major |

A vector, a `std::array`, a `std::span` and a raw pointer are all the *same*
rung: `flat()` turns each into a `std::span` and hands it to the one shape-taking
`as_view`. The only difference is that a range's size is checked against the
shape; a pointer's span is sized *from* the shape, so it passes the same check.

```cpp
plan->eval(eigenA, eigenB);                             // Eigen::Matrix
plan->eval(mdA, mdB);                                   // buffer; .as_mdspan<2>()
plan->eval(flat(vecA, {2, 3}), flat(vecB, {3, 4}));     // std::vector<double>
plan->eval(flat(pA, {2, 3}), flat(pB, {3, 4}),
           into(pC, {2, 4}));                           // writes through pC
```

A mixed call is a compile-time error carrying that sentence, not a runtime
`errc`. A range that is not both contiguous and sized has no strides to describe
and no size to check a shape against, and is refused.

`into(out)` is the explicit, zero-allocation form and works for every kind: name
it and `eval` answers `result<void>` instead. The mdspan rung hands back the
buffer rather than an mdspan because a runtime subscript's rank is in no type —
`as_mdspan<R>()` answers `rank_mismatch` if `R` is not the rank it produced.

## The runtime API

```cpp
const auto plan = einsum::rt::plan("bij,bjk->bik");   // result<Plan>

auto bound = plan->bind(a, b, into(c));               // result<Bound<double>>
bound->eval();                                        // noexcept, allocates nothing
bound->eval(a2, b2, into(c2));                        // same shapes, new pointers

plan->bind(a, b, into(c), std::span<double>{scratch}); // borrow the workspace too
```

- **`bind`** converts every operand, lowers the plan against their extents and
  strides, and rents the scratch. Everything that can fail, fails here. It needs
  `into(out)`: an object you keep and the buffer it writes into cannot both be
  the return value.
- **`eval()`** is `noexcept` and allocates nothing. Every buffer it could want is
  an offset into the workspace, sized when the plan was bound.
- **`eval(args...)`** rebinds pointers only. Extents *and* strides have to match
  what was bound: a `Geometry` is a set of offsets, and an operand that moved its
  rows is a different one however alike its shape looks.
- **`operator()` / `eval` on the `Plan`** is bind-and-run in one call. With
  `into(out)` the workspace is the one allocation and it is gone again by the
  time it returns; without it, the result is allocated and handed back in the
  operands' own kind.
- **`to_matrix(ins...)`** is the convenience form for a rank-≤2 result: it
  allocates the answer and the workspace, and says so.
- **`workspace_data()`**, **`scratch_bytes()`**, **`output_shape()`** describe
  what was rented.

### The zero-allocation guarantee, exactly

`Bound::eval()` performs no allocation *of its own* — that is checked by
`tests/tests_noalloc.cpp`, which replaces `operator new` and counts. Eigen's GEMM
blocking buffers are a separate matter: below `EIGEN_STACK_ALLOCATION_LIMIT`
(128 KB) they live on the stack, and above it Eigen calls `std::malloc` itself.
For very large products, then, einsum allocates nothing and Eigen may.

## The compile-time API

```cpp
auto ein = einsum::einsum<"ij,jk->ik">(a, b);
ein.eval();

ein.result();            // the operands' own kind
ein.get_result();        // const std::array<T, output_size>&
ein.get_result_span();   // std::mdspan<const T, extents_type>
ein.result_matrix();     // Eigen::Map, rank <= 2, no copy

decltype(ein)::output_size;    // std::size_t
decltype(ein)::scratch_size;   // 0 for a plain packed matmul
decltype(ein)::extents_type;   // std::extents<size_t, 2, 4>
```

Operands must have their whole layout in their type: an mdspan with all-static
extents and `layout_right`/`layout_left`, or a fixed-size `Eigen::Matrix`. The
object holds views of them, so it must not outlive them.

Every refusal is a `static_assert` carrying the same sentence the runtime error
would have carried, generated from the same table the codes are:

```
error: static assertion failed: einsum<"...">: one label is bound to two different extents
```

`tests/compile_fail_*.cpp` are built on purpose by ctest and pass when the
compiler refuses.

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
- **Contraction order is the subscript's.** `"ij,jk,kl->il"` associates left to
  right; this library does not reorder, because it has no cost model that would
  justify doing so.
- **Labels** are `[a-zA-Z]`. Whitespace between them is ignored.
- **No ellipsis and no broadcasting.** `...` is refused by name.

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
| `ellipsis_unsupported` | `...` is not supported: name every axis |
| `empty_operand` | an operand between the commas has no labels |
| `no_operands` | the subscript names no operands |
| `too_many_operands` | more operands than `kMaxOperands` |
| `rank_too_high` | an operand has more labels than `kMaxRank` |
| `unknown_output_label` | the output names a label that no operand has |
| `repeated_output_label` | the output repeats a label |
| `operand_count_mismatch` | the subscript and the call disagree on how many operands there are |
| `rank_mismatch` | an operand's rank differs from the number of labels it was given |
| `extent_conflict` | one label is bound to two different extents |
| `output_mismatch` | the output's rank or extents are not the ones the subscript implies |
| `size_mismatch` | the memory is smaller than the shape it was given |
| `workspace_too_small` | the borrowed workspace is smaller than the plan needs |
| `not_matrix` | `to_matrix()` needs a result of rank 2 or less |

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
              CtEinsum<S, Ops...>                                    Bound<T>
        Owned<T, std::array<T, N>>              Owned<T, std::vector<T>>, or the caller's view
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

Covers the header-only library: `einsum<"...">`, `Plan`, `Bound`, the views, the
lowering and the kernels. It still needs Boost, Eigen and mdspan on the include
path. It does **not** cover `einsum::rt::plan()` — that grammar is compiled into
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
