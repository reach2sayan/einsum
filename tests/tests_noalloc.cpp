// The zero-allocation claim, checked by counting.  Its own executable because
// replacing operator new is a program-wide decision, and a suite that also
// measures something else would be measuring Boost.Test as well.
//
// What this proves: einsum allocates nothing after bind() returns.  What it
// cannot prove: that Eigen allocates nothing, since Eigen's blocking buffers go
// through std::malloc directly (below EIGEN_STACK_ALLOCATION_LIMIT they are not
// heap at all).  That caveat is in the README beside the guarantee.
#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#define BOOST_TEST_MODULE EinsumNoAllocSuite
#include <boost/test/included/unit_test.hpp>

namespace {
bool g_counting = false;
long g_allocations = 0;

struct Counting {
  Counting() noexcept {
    g_allocations = 0;
    g_counting = true;
  }
  Counting(const Counting &) = delete;
  Counting &operator=(const Counting &) = delete;
  ~Counting() { g_counting = false; }
};
} // namespace

// -fno-exceptions, so a failed allocation aborts rather than throwing; nothing
// in this suite is meant to run out of memory anyway.
void *operator new(const std::size_t size) {
  if (g_counting) {
    ++g_allocations;
  }
  void *memory = std::malloc(size == 0 ? 1 : size);
  if (memory == nullptr) {
    std::fputs("tests_noalloc: out of memory\n", stderr);
    std::abort();
  }
  return memory;
}
void *operator new[](const std::size_t size) { return ::operator new(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

namespace es = einsum;

namespace {

[[nodiscard]] Eigen::MatrixXd sample(const es::index_t n, const int seed) {
  Eigen::MatrixXd m(n, n);
  for (es::index_t r = 0; r < n; ++r) {
    for (es::index_t c = 0; c < n; ++c) {
      m(r, c) = static_cast<double>((r * 5 + c * 3 + seed) % 9) - 4.0;
    }
  }
  return m;
}

void check_no_alloc(const char *subscript, const es::index_t n) {
  const auto plan = es::rt::plan(subscript);
  BOOST_REQUIRE(plan.has_value());
  const Eigen::MatrixXd a = sample(n, 1);
  const Eigen::MatrixXd b = sample(n, 4);
  Eigen::MatrixXd out(n, n);
  auto bound = plan->bind(a, b, es::into(out));
  BOOST_REQUIRE(bound.has_value());

  const double *workspace_before = bound->workspace_data().data();
  {
    const Counting counting;
    bound->eval();
  }
  BOOST_TEST_CONTEXT(subscript << " at " << n) {
    BOOST_CHECK_EQUAL(g_allocations, 0);
  }

  // Rebinding moves pointers and nothing else, so it cannot allocate either --
  // and the workspace has to be the same one afterwards.
  const Eigen::MatrixXd a2 = sample(n, 8);
  const Eigen::MatrixXd b2 = sample(n, 2);
  Eigen::MatrixXd out2(n, n);
  {
    const Counting counting;
    const auto ok = bound->eval(a2, b2, es::into(out2));
    BOOST_REQUIRE(ok.has_value());
  }
  BOOST_CHECK_EQUAL(g_allocations, 0);
  BOOST_CHECK_EQUAL(bound->workspace_data().data(), workspace_before);
  BOOST_CHECK_LT((out2 - a2 * b2).cwiseAbs().maxCoeff(), 1e-9);
}

} // namespace

BOOST_AUTO_TEST_CASE(NoAlloc_MatmulAtSeveralSizes) {
  for (const es::index_t n : {4, 16, 64}) {
    check_no_alloc("ij,jk->ik", n);
  }
}

BOOST_AUTO_TEST_CASE(NoAlloc_APackedContraction) {
  // "ikj,jkl->il" has to pack its left slab, so it uses the workspace rather
  // than mapping straight through -- which is the case worth counting.
  const auto plan = es::rt::plan("ikj,jkl->il");
  BOOST_REQUIRE(plan.has_value());
  std::vector<double> a(2 * 4 * 3, 1.5);
  std::vector<double> b(3 * 4 * 5, 0.5);
  std::vector<double> out(2 * 5, 0.0);
  auto bound = plan->bind(es::flat(a, {2, 4, 3}), es::flat(b, {3, 4, 5}),
                          es::into(out, {2, 5}));
  BOOST_REQUIRE(bound.has_value());
  BOOST_CHECK_GT(bound->scratch_bytes(), 0U);
  {
    const Counting counting;
    bound->eval();
  }
  BOOST_CHECK_EQUAL(g_allocations, 0);
  for (const double x : out) {
    BOOST_CHECK_LT(std::abs(x - 9.0), 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(NoAlloc_BorrowedWorkspaceAllocatesNothingAtAll) {
  const auto plan = es::rt::plan("ij,jk,kl->il");
  BOOST_REQUIRE(plan.has_value());
  const Eigen::MatrixXd a = sample(8, 1);
  const Eigen::MatrixXd b = sample(8, 2);
  const Eigen::MatrixXd c = sample(8, 3);
  Eigen::MatrixXd out(8, 8);
  std::vector<double> workspace(4096);

  auto bound = plan->bind(a, b, c, es::into(out), std::span<double>{workspace});
  BOOST_REQUIRE(bound.has_value());
  BOOST_CHECK_EQUAL(bound->workspace_data().data(), workspace.data());
  {
    const Counting counting;
    bound->eval();
  }
  BOOST_CHECK_EQUAL(g_allocations, 0);
  BOOST_CHECK_LT((out - a * b * c).cwiseAbs().maxCoeff(), 1e-9);
}
