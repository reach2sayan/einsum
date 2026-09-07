// The allocation claim, checked by counting.  Its own executable because
// replacing operator new is a program-wide decision, and a suite that also
// measured something else would be measuring Boost.Test as well.
//
// What this proves: once an object's scratch cache is warm, a repeated call
// allocates nothing through operator new beyond the result it hands back.
//
// What it cannot prove: that Eigen allocates nothing.  Eigen's buffers go
// through its own aligned_malloc -- std::malloc, not operator new -- so an
// Eigen result does not appear in these counts at all, and neither do the GEMM
// blocking buffers (below EIGEN_STACK_ALLOCATION_LIMIT they are not heap).
// That caveat is in the README beside the guarantee.
#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <cstdio>
#include <experimental/mdspan>
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
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

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

} // namespace

// Every count is taken inside the counting scope and asserted outside it,
// because Boost.Test's own reporting allocates.
BOOST_AUTO_TEST_CASE(NoAlloc_WarmEigenCallAllocatesNothingWeCanSee) {
  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Eigen::MatrixXd a = sample(16, 1);
  const Eigen::MatrixXd b = sample(16, 4);
  BOOST_REQUIRE((*plan)(a, b).has_value());

  long counted = -1;
  bool ok = false;
  {
    const Counting counting;
    const auto again = (*plan)(a, b);
    ok = again.has_value();
    counted = g_allocations;
  }
  BOOST_REQUIRE(ok);
  BOOST_CHECK_EQUAL(counted, 0);
}

// A contraction that has to pack a slab uses the scratch cache; once warm, the
// cache is neither grown nor freed, so two consecutive calls cost the same.
BOOST_AUTO_TEST_CASE(NoAlloc_PackedContractionReusesItsScratch) {
  const auto plan = es::einsum("ikj,jkl->il");
  BOOST_REQUIRE(plan.has_value());
  const std::vector<std::vector<std::vector<double>>> x(
      2, std::vector<std::vector<double>>(4, std::vector<double>(3, 1.5)));
  const std::vector<std::vector<std::vector<double>>> y(
      3, std::vector<std::vector<double>>(4, std::vector<double>(5, 0.5)));
  BOOST_REQUIRE((*plan)(x, y).has_value());

  long first = -1;
  long second = -2;
  bool ok = false;
  {
    const Counting counting;
    const auto a1 = (*plan)(x, y);
    first = g_allocations;
    g_allocations = 0;
    const auto a2 = (*plan)(x, y);
    second = g_allocations;
    ok = a1.has_value() && a2.has_value();
  }
  BOOST_REQUIRE(ok);
  // Whatever the nest result costs, it costs the same twice: the scratch is not
  // among the allocations.
  BOOST_CHECK_EQUAL(first, second);
}

// Static-extent mdspan operands on the compile-time path: every extent is in a
// type, so the geometry is a constant, the scratch is an array in the call's own
// frame, and the result is an mdarray over a std::array.  Nothing is allocated
// at all -- not even the result.
BOOST_AUTO_TEST_CASE(NoAlloc_StaticMdspanCallAllocatesNothing) {
  const std::vector<double> a(16, 1.0);
  const std::vector<double> b(16, 2.0);
  const std::mdspan<const double, std::extents<std::size_t, 4, 4>> ma{a.data()};
  const std::mdspan<const double, std::extents<std::size_t, 4, 4>> mb{b.data()};
  const auto e = es::einsum<"ij,jk->ik">();
  BOOST_REQUIRE(e(ma, mb).has_value());

  long counted = -1;
  bool ok = false;
  {
    const Counting counting;
    const auto again = e(ma, mb);
    ok = again.has_value();
    counted = g_allocations;
  }
  BOOST_REQUIRE(ok);
  BOOST_CHECK_EQUAL(counted, 0);
}

// A warm runtime call of the same shape reuses both caches -- the scratch and
// the geometry -- so it allocates only what it hands back.
BOOST_AUTO_TEST_CASE(NoAlloc_WarmSameShapeCallReusesTheGeometry) {
  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Eigen::MatrixXd a = sample(12, 1);
  const Eigen::MatrixXd b = sample(12, 3);
  BOOST_REQUIRE((*plan)(a, b).has_value());

  long counted = -1;
  bool ok = false;
  {
    const Counting counting;
    const auto again = (*plan)(a, b);
    ok = again.has_value();
    counted = g_allocations;
  }
  BOOST_REQUIRE(ok);
  // The Eigen result goes through Eigen's own malloc, so nothing reaches
  // operator new at all.
  BOOST_CHECK_EQUAL(counted, 0);
}

// The compile-time object over an Eigen result: same guarantee, and building
// the object itself never allocates.
BOOST_AUTO_TEST_CASE(NoAlloc_StaticObjectWarmCall) {
  const auto e = es::einsum<"ij,jk->ik">();
  const Eigen::MatrixXd a = sample(8, 1);
  const Eigen::MatrixXd b = sample(8, 2);
  BOOST_REQUIRE(e(a, b).has_value());

  long counted = -1;
  bool ok = false;
  {
    const Counting counting;
    const auto again = e(a, b);
    ok = again.has_value();
    counted = g_allocations;
  }
  BOOST_REQUIRE(ok);
  BOOST_CHECK_EQUAL(counted, 0);
}
