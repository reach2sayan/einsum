// What the three ways of writing one contraction cost: the loop you would have
// written, the compile-time einsum, the runtime one, and Eigen called directly.
// The families are generated over one size list, so adding a size adds it
// everywhere and no family can quietly fall behind the others.
#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <benchmark/benchmark.h>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include <vector>

namespace es = einsum;

#define EINSUM_SIZES (2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(16)(32)(64)
#define EINSUM_SCALARS (int)(double)

namespace {

template <typename T> std::vector<T> filled(const std::size_t n, const int seed) {
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

} // namespace

// --- the families ------------------------------------------------------------
// BENCHMARK() stringifies its argument, which suppresses macro expansion, so
// the name is built once and passed through one more level of indirection --
// otherwise every family registers under the literal text of the CAT.
#define EINSUM_BENCH_NAME(prefix, product)                                              \
  BOOST_PP_CAT(BOOST_PP_CAT(prefix, BOOST_PP_SEQ_ELEM(0, product)),                     \
               BOOST_PP_CAT(_, BOOST_PP_SEQ_ELEM(1, product)))
#define EINSUM_REGISTER(name) BENCHMARK(name)
#define EINSUM_BENCH_NAIVE(r, product)                                                  \
  static void EINSUM_BENCH_NAME(BM_naive_, product)(             \
      benchmark::State & state) {                                                       \
    using T = BOOST_PP_SEQ_ELEM(0, product);                                            \
    constexpr es::index_t n = BOOST_PP_SEQ_ELEM(1, product);                            \
    const auto a = filled<T>(n * n, 1);                                                 \
    const auto b = filled<T>(n * n, 5);                                                 \
    std::vector<T> c(static_cast<std::size_t>(n * n));                                  \
    for (auto _ : state) {                                                              \
      naive_matmul<T, n>(a.data(), b.data(), c.data());                                 \
      benchmark::DoNotOptimize(c.data());                                               \
      benchmark::ClobberMemory();                                                       \
    }                                                                                   \
  }                                                                                     \
  EINSUM_REGISTER(EINSUM_BENCH_NAME(BM_naive_, product));

#define EINSUM_BENCH_NAIVE_OPT(r, product)                                              \
  static void EINSUM_BENCH_NAME(BM_naive_opt_, product)(             \
      benchmark::State & state) {                                                       \
    using T = BOOST_PP_SEQ_ELEM(0, product);                                            \
    constexpr es::index_t n = BOOST_PP_SEQ_ELEM(1, product);                            \
    const auto a = filled<T>(n * n, 1);                                                 \
    const auto b = filled<T>(n * n, 5);                                                 \
    std::vector<T> c(static_cast<std::size_t>(n * n));                                  \
    for (auto _ : state) {                                                              \
      naive_matmul_opt<T, n>(a.data(), b.data(), c.data());                             \
      benchmark::DoNotOptimize(c.data());                                               \
      benchmark::ClobberMemory();                                                       \
    }                                                                                   \
  }                                                                                     \
  EINSUM_REGISTER(EINSUM_BENCH_NAME(BM_naive_opt_, product));

#define EINSUM_BENCH_CT(r, product)                                                     \
  static void EINSUM_BENCH_NAME(BM_einsum_ct_, product)(             \
      benchmark::State & state) {                                                       \
    using T = BOOST_PP_SEQ_ELEM(0, product);                                            \
    constexpr std::size_t n = BOOST_PP_SEQ_ELEM(1, product);                            \
    const auto a = filled<T>(n * n, 1);                                                 \
    const auto b = filled<T>(n * n, 5);                                                 \
    std::mdspan<const T, std::extents<std::size_t, n, n>> ma{a.data()};                 \
    std::mdspan<const T, std::extents<std::size_t, n, n>> mb{b.data()};                 \
    auto ein = es::einsum<"ij,jk->ik">(ma, mb);                                         \
    for (auto _ : state) {                                                              \
      ein.eval();                                                                       \
      benchmark::DoNotOptimize(ein.get_result().data());                                \
      benchmark::ClobberMemory();                                                       \
    }                                                                                   \
  }                                                                                     \
  EINSUM_REGISTER(EINSUM_BENCH_NAME(BM_einsum_ct_, product));

#define EINSUM_BENCH_RT(r, product)                                                     \
  static void EINSUM_BENCH_NAME(BM_einsum_rt_, product)(             \
      benchmark::State & state) {                                                       \
    using T = BOOST_PP_SEQ_ELEM(0, product);                                            \
    constexpr es::index_t n = BOOST_PP_SEQ_ELEM(1, product);                            \
    const auto a = filled<T>(static_cast<std::size_t>(n * n), 1);                       \
    const auto b = filled<T>(static_cast<std::size_t>(n * n), 5);                       \
    std::vector<T> c(static_cast<std::size_t>(n * n));                                  \
    const auto plan = es::rt::plan("ij,jk->ik");                                        \
    auto bound = plan->bind(es::flat(a, {n, n}), es::flat(b, {n, n}),               \
                            es::into(c, {n, n}));                                       \
    for (auto _ : state) {                                                              \
      bound->eval();                                                                    \
      benchmark::DoNotOptimize(c.data());                                               \
      benchmark::ClobberMemory();                                                       \
    }                                                                                   \
  }                                                                                     \
  EINSUM_REGISTER(EINSUM_BENCH_NAME(BM_einsum_rt_, product));

#define EINSUM_BENCH_EIGEN_FIXED(r, product)                                            \
  static void EINSUM_BENCH_NAME(BM_eigen_fixed_, product)(             \
      benchmark::State & state) {                                                       \
    using T = BOOST_PP_SEQ_ELEM(0, product);                                            \
    constexpr int n = BOOST_PP_SEQ_ELEM(1, product);                                    \
    Eigen::Matrix<T, n, n, Eigen::RowMajor> a;                                          \
    Eigen::Matrix<T, n, n, Eigen::RowMajor> b;                                          \
    Eigen::Matrix<T, n, n, Eigen::RowMajor> c;                                          \
    a.setOnes();                                                                        \
    b.setOnes();                                                                        \
    for (auto _ : state) {                                                              \
      c.noalias() = a * b;                                                              \
      benchmark::DoNotOptimize(c.data());                                               \
      benchmark::ClobberMemory();                                                       \
    }                                                                                   \
  }                                                                                     \
  EINSUM_REGISTER(EINSUM_BENCH_NAME(BM_eigen_fixed_, product));

#define EINSUM_BENCH_EIGEN_DYNAMIC(r, product)                                          \
  static void EINSUM_BENCH_NAME(BM_eigen_dyn_, product)(             \
      benchmark::State & state) {                                                       \
    using T = BOOST_PP_SEQ_ELEM(0, product);                                            \
    constexpr int n = BOOST_PP_SEQ_ELEM(1, product);                                    \
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a(n, n);          \
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(n, n);          \
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c(n, n);          \
    a.setOnes();                                                                        \
    b.setOnes();                                                                        \
    for (auto _ : state) {                                                              \
      c.noalias() = a * b;                                                              \
      benchmark::DoNotOptimize(c.data());                                               \
      benchmark::ClobberMemory();                                                       \
    }                                                                                   \
  }                                                                                     \
  EINSUM_REGISTER(EINSUM_BENCH_NAME(BM_eigen_dyn_, product));

BOOST_PP_SEQ_FOR_EACH_PRODUCT(EINSUM_BENCH_NAIVE, (EINSUM_SCALARS)(EINSUM_SIZES))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EINSUM_BENCH_NAIVE_OPT, (EINSUM_SCALARS)(EINSUM_SIZES))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EINSUM_BENCH_CT, (EINSUM_SCALARS)(EINSUM_SIZES))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EINSUM_BENCH_RT, (EINSUM_SCALARS)(EINSUM_SIZES))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EINSUM_BENCH_EIGEN_FIXED, (EINSUM_SCALARS)(EINSUM_SIZES))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EINSUM_BENCH_EIGEN_DYNAMIC, (EINSUM_SCALARS)(EINSUM_SIZES))

// --- the shapes a plain matmul does not cover --------------------------------
// A batch, and a unary permutation: the two lowerings that are not one GEMM.
static void BM_einsum_batched_double_16(benchmark::State &state) {
  constexpr std::size_t b = 8;
  constexpr std::size_t n = 16;
  const auto x = filled<double>(b * n * n, 1);
  const auto y = filled<double>(b * n * n, 3);
  std::mdspan<const double, std::extents<std::size_t, b, n, n>> mx{x.data()};
  std::mdspan<const double, std::extents<std::size_t, b, n, n>> my{y.data()};
  auto ein = es::einsum<"bij,bjk->bik">(mx, my);
  for (auto _ : state) {
    ein.eval();
    benchmark::DoNotOptimize(ein.get_result().data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_einsum_batched_double_16);

static void BM_einsum_transpose_double_64(benchmark::State &state) {
  constexpr std::size_t n = 64;
  const auto x = filled<double>(n * n, 1);
  std::mdspan<const double, std::extents<std::size_t, n, n>> mx{x.data()};
  auto ein = es::einsum<"ij->ji">(mx);
  for (auto _ : state) {
    ein.eval();
    benchmark::DoNotOptimize(ein.get_result().data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_einsum_transpose_double_64);

BENCHMARK_MAIN();
