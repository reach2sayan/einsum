// What the three ways of writing one contraction cost: the loop you would have
// written, the compile-time einsum, the runtime one, and Eigen called directly.
//
// Two registration styles, because the families split in two.  A family whose
// size is a genuine runtime value is one function taking state.range(0) and a
// list of ->Arg(); a family that needs the size in a type is a template that
// mp_for_each instantiates and RegisterBenchmark names.  Either way the sizes
// come from one list, so no family can quietly fall behind the others.
#include <unsupported/Eigen/CXX11/Tensor>

#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <benchmark/benchmark.h>

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <format>
#include <string>
#include <vector>

namespace es = einsum;
namespace mp = boost::mp11;

namespace {

using Sizes =
    mp::mp_list_c<int, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 64>;

template <typename T>
std::vector<T> filled(const std::size_t n, const int seed) {
  std::vector<T> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<T>((i * 7 + static_cast<std::size_t>(seed)) % 13);
  }
  return out;
}

// The straightforward triple loop, and the same one with the accumulator lifted
// out of the inner iteration -- the two things a reader would write by hand.
template <typename T, es::index_t N>
void naive_matmul(const T *a, const T *b, T *c) noexcept {
  for (es::index_t i = 0; i < N; ++i) {
    for (es::index_t k = 0; k < N; ++k) {
      c[i * N + k] = T{};
      for (es::index_t j = 0; j < N; ++j) {
        c[i * N + k] += a[i * N + j] * b[j * N + k];
      }
    }
  }
}

template <typename T, es::index_t N>
void naive_matmul_opt(const T *a, const T *b, T *c) noexcept {
  for (es::index_t i = 0; i < N; ++i) {
    for (es::index_t k = 0; k < N; ++k) {
      T acc{};
      for (es::index_t j = 0; j < N; ++j) {
        acc += a[i * N + j] * b[j * N + k];
      }
      c[i * N + k] = acc;
    }
  }
}

// --- the families whose size is in a type ------------------------------------
template <typename T, int N> void bm_naive(benchmark::State &state) {
  const auto a = filled<T>(N * N, 1);
  const auto b = filled<T>(N * N, 5);
  std::vector<T> c(static_cast<std::size_t>(N) * N);
  for (auto _ : state) {
    naive_matmul<T, N>(a.data(), b.data(), c.data());
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
}

template <typename T, int N> void bm_naive_opt(benchmark::State &state) {
  const auto a = filled<T>(N * N, 1);
  const auto b = filled<T>(N * N, 5);
  std::vector<T> c(static_cast<std::size_t>(N) * N);
  for (auto _ : state) {
    naive_matmul_opt<T, N>(a.data(), b.data(), c.data());
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
}

template <typename T, int N> void bm_einsum_ct(benchmark::State &state) {
  constexpr std::size_t n = static_cast<std::size_t>(N);
  const auto a = filled<T>(n * n, 1);
  const auto b = filled<T>(n * n, 5);
  const std::mdspan<const T, std::extents<std::size_t, n, n>> ma{a.data()};
  const std::mdspan<const T, std::extents<std::size_t, n, n>> mb{b.data()};
  const auto ein = es::einsum<"ij,jk->ik">();
  for (auto _ : state) {
    auto out = ein(ma, mb);
    benchmark::DoNotOptimize(out->data());
    benchmark::ClobberMemory();
  }
}

// Fixed-size Eigen operands, which is where the compile-time path can do what
// the runtime one cannot: every extent is in a type, so the whole lowering is a
// constant and the product is Eigen's unrolled fixed-size one.
template <typename T, int N> void bm_einsum_ct_eigen(benchmark::State &state) {
  Eigen::Matrix<T, N, N> a; // Eigen's default order, column-major
  Eigen::Matrix<T, N, N> b;
  a.setOnes();
  b.setOnes();
  static_assert(
      es::StaticEinsum<"ij,jk->ik">::unrolls<Eigen::Matrix<T, N, N>,
                                             Eigen::Matrix<T, N, N>>(),
      "this family exists to measure the unrolled product");
  const auto ein = es::einsum<"ij,jk->ik">();
  for (auto _ : state) {
    auto out = ein(a, b);
    benchmark::DoNotOptimize(out->data());
    benchmark::ClobberMemory();
  }
}

template <typename T, int N> void bm_eigen_fixed(benchmark::State &state) {
  Eigen::Matrix<T, N, N, Eigen::RowMajor> a;
  Eigen::Matrix<T, N, N, Eigen::RowMajor> b;
  Eigen::Matrix<T, N, N, Eigen::RowMajor> c;
  a.setOnes();
  b.setOnes();
  for (auto _ : state) {
    c.noalias() = a * b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
}

// --- the families whose size is only a number --------------------------------
template <typename T> void bm_einsum_rt(benchmark::State &state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a(n, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(n, n);
  a.setOnes();
  b.setOnes();
  const auto plan = es::einsum("ij,jk->ik");
  for (auto _ : state) {
    auto out = (*plan)(a, b);
    benchmark::DoNotOptimize(out->data());
    benchmark::ClobberMemory();
  }
}

template <typename T> void bm_eigen_dyn(benchmark::State &state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a(n, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(n, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c(n, n);
  a.setOnes();
  b.setOnes();
  for (auto _ : state) {
    c.noalias() = a * b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
}

// --- registration
// ------------------------------------------------------------- The names are
// the ones the old BOOST_PP families registered under, so a before/after
// comparison still lines up.
template <typename T> constexpr const char *scalar_name() {
  return std::same_as<T, int> ? "int" : "double";
}

// A dynamic family is one function over every size, which is what ->Arg() is
// for.
template <typename T, void (*Fn)(benchmark::State &)>
void register_dynamic(const char *prefix) {
  auto *entry = benchmark::RegisterBenchmark(
      std::format("{}{}", prefix, scalar_name<T>()), Fn);
  mp::mp_for_each<Sizes>(
      [entry](auto size) { entry->Arg(decltype(size)::value); });
}

// The four fixed-size families share one walk over the sizes: mp_for_each hands
// each one to RegisterBenchmark as a type, which is the only way a family whose
// N is a template argument can be generated at all.
template <typename T> void register_all() {
  mp::mp_for_each<Sizes>([](auto size) {
    constexpr int n = decltype(size)::value;
    const auto named = [n](const char *prefix) {
      return std::format("{}{}_{}", prefix, scalar_name<T>(), n);
    };
    benchmark::RegisterBenchmark(named("BM_naive_"), bm_naive<T, n>);
    benchmark::RegisterBenchmark(named("BM_naive_opt_"), bm_naive_opt<T, n>);
    benchmark::RegisterBenchmark(named("BM_einsum_ct_"), bm_einsum_ct<T, n>);
    benchmark::RegisterBenchmark(named("BM_einsum_ct_eigen_"),
                                 bm_einsum_ct_eigen<T, n>);
    benchmark::RegisterBenchmark(named("BM_eigen_fixed_"),
                                 bm_eigen_fixed<T, n>);
  });
  register_dynamic<T, bm_einsum_rt<T>>("BM_einsum_rt_");
  register_dynamic<T, bm_eigen_dyn<T>>("BM_eigen_dyn_");
}

// --- the shapes a plain matmul does not cover --------------------------------
// A batch, and a unary permutation: the two lowerings that are not one GEMM.
// An Eigen Tensor, so a rank-3 result is still an owning family the kernels
// write straight into: no gather in, no scatter out, and one allocation for the
// result rather than one per row of a nest.
void bm_einsum_batched(benchmark::State &state) {
  constexpr int b = 8;
  constexpr int n = 16;
  Eigen::Tensor<double, 3, Eigen::RowMajor> x(b, n, n);
  Eigen::Tensor<double, 3, Eigen::RowMajor> y(b, n, n);
  x.setConstant(1.0);
  y.setConstant(3.0);
  const auto ein = es::einsum<"bij,bjk->bik">();
  for (auto _ : state) {
    auto out = ein(x, y);
    benchmark::DoNotOptimize(out->data());
    benchmark::ClobberMemory();
  }
}

void bm_einsum_transpose(benchmark::State &state) {
  constexpr int n = 64;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(n,
                                                                           n);
  x.setOnes();
  const auto ein = es::einsum<"ij->ji">();
  for (auto _ : state) {
    auto out = ein(x);
    benchmark::DoNotOptimize(out->data());
    benchmark::ClobberMemory();
  }
}

} // namespace

int main(int argc, char **argv) {
  register_all<int>();
  register_all<double>();
  benchmark::RegisterBenchmark("BM_einsum_batched_double_16",
                               bm_einsum_batched);
  benchmark::RegisterBenchmark("BM_einsum_transpose_double_64",
                               bm_einsum_transpose);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
