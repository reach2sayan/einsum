// The runtime path: einsum(subscript) meeting operands whose shapes nothing
// knew until now.  The matmul matrix below is generated -- one case per operand
// family per size -- because the interesting differences between the families
// are how their memory is reached, and a hand-written case per pair would be
// three copies of one loop.
#include <unsupported/Eigen/CXX11/Tensor>

#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <cmath>
#include <experimental/mdarray>
#include <experimental/mdspan>
#include <vector>

#define BOOST_TEST_MODULE EinsumRuntimeSuite
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/included/unit_test.hpp>

namespace es = einsum;
using es::errc;
using es::index_t;

namespace {

using Matrix = Eigen::MatrixXd;
using Nest = std::vector<std::vector<double>>;

[[nodiscard]] Matrix sample(const index_t rows, const index_t cols,
                            const int seed) {
  Matrix m(rows, cols);
  for (index_t r = 0; r < rows; ++r) {
    for (index_t c = 0; c < cols; ++c) {
      m(r, c) = static_cast<double>((r * 7 + c * 3 + seed) % 11) - 5.0;
    }
  }
  return m;
}

// --- one holder per family
// ---------------------------------------------------- Each fills itself from a
// matrix, hands back something a call takes, and reads back as a matrix.
// Nothing else about them is alike, which is the point.
struct EigenFamily {
  Matrix m;
  explicit EigenFamily(const Matrix &src) : m{src} {}
  [[nodiscard]] const Matrix &operand() const { return m; }
  [[nodiscard]] static Matrix read(const Matrix &r) { return r; }
};

struct NestFamily {
  Nest m;
  explicit NestFamily(const Matrix &src)
      : m(static_cast<std::size_t>(src.rows()),
          std::vector<double>(static_cast<std::size_t>(src.cols()))) {
    for (index_t r = 0; r < src.rows(); ++r) {
      for (index_t c = 0; c < src.cols(); ++c) {
        m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = src(r, c);
      }
    }
  }
  [[nodiscard]] const Nest &operand() const { return m; }
  [[nodiscard]] static Matrix read(const Nest &r) {
    Matrix out(static_cast<index_t>(r.size()),
               static_cast<index_t>(r[0].size()));
    for (std::size_t i = 0; i < r.size(); ++i) {
      for (std::size_t j = 0; j < r[i].size(); ++j) {
        out(static_cast<index_t>(i), static_cast<index_t>(j)) = r[i][j];
      }
    }
    return out;
  }
};

// Row-major storage described by an mdspan: a view family, so the result comes
// back as a nest of vectors rather than as a view of anything.
struct MdspanFamily {
  std::vector<double> buffer;
  std::size_t rows;
  std::size_t cols;
  explicit MdspanFamily(const Matrix &src)
      : buffer(static_cast<std::size_t>(src.size())),
        rows{static_cast<std::size_t>(src.rows())},
        cols{static_cast<std::size_t>(src.cols())} {
    for (index_t r = 0; r < src.rows(); ++r) {
      for (index_t c = 0; c < src.cols(); ++c) {
        buffer[static_cast<std::size_t>(r) * cols +
               static_cast<std::size_t>(c)] = src(r, c);
      }
    }
  }
  [[nodiscard]] auto operand() const {
    return std::mdspan<const double, std::dextents<std::size_t, 2>>{
        buffer.data(), rows, cols};
  }
  [[nodiscard]] static Matrix read(const Nest &r) {
    return NestFamily::read(r);
  }
};

const std::vector<index_t> kSizes{2, 3, 4, 8, 16, 33};

template <typename Family> void check_matmul(const index_t n) {
  const Matrix a = sample(n, n, 1);
  const Matrix b = sample(n, n, 5);
  const Family left{a};
  const Family right{b};

  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const auto got = (*plan)(left.operand(), right.operand());
  BOOST_REQUIRE_MESSAGE(got.has_value(), es::message(got.error().code));
  BOOST_CHECK_LT((Family::read(*got) - a * b).cwiseAbs().maxCoeff(), 1e-12);
}

} // namespace

BOOST_DATA_TEST_CASE(RtMatmul_EigenFamily, boost::unit_test::data::make(kSizes),
                     n) {
  check_matmul<EigenFamily>(n);
}
BOOST_DATA_TEST_CASE(RtMatmul_NestFamily, boost::unit_test::data::make(kSizes),
                     n) {
  check_matmul<NestFamily>(n);
}
BOOST_DATA_TEST_CASE(RtMatmul_MdspanFamily,
                     boost::unit_test::data::make(kSizes), n) {
  check_matmul<MdspanFamily>(n);
}

// --- the type a call answers
// --------------------------------------------------
BOOST_AUTO_TEST_CASE(RtResult_FamilyDecidesTheType) {
  const auto plan = es::einsum("ij,jk->ik");
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);
  const auto eigen = (*plan)(a, b);
  static_assert(std::same_as<std::remove_cvref_t<decltype(*eigen)>, Matrix>);
  BOOST_REQUIRE(eigen.has_value());
  BOOST_CHECK_LT((*eigen - a * b).cwiseAbs().maxCoeff(), 1e-12);

  // A view cannot own a result, so the view family answers a nest of vectors.
  const MdspanFamily left{a};
  const MdspanFamily right{b};
  const auto viewed = (*plan)(left.operand(), right.operand());
  static_assert(std::same_as<std::remove_cvref_t<decltype(*viewed)>, Nest>);
  BOOST_REQUIRE(viewed.has_value());
  BOOST_CHECK_LT((NestFamily::read(*viewed) - a * b).cwiseAbs().maxCoeff(),
                 1e-12);
}

// In place is a move, not an API: there is no output parameter on this path.
BOOST_AUTO_TEST_CASE(RtResult_InPlaceIsAMove) {
  const auto plan = es::einsum("ij,jk->ik");
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);
  Matrix out(1, 1);
  out = *(*plan)(a, b);
  BOOST_CHECK_LT((out - a * b).cwiseAbs().maxCoeff(), 1e-12);
}

// --- mixed ranks, which same-family (not same-type) is what allows
// ------------
BOOST_AUTO_TEST_CASE(RtMixedRank_MatrixTimesVector) {
  const auto plan = es::einsum("ij,j->i");
  BOOST_REQUIRE(plan.has_value());
  const Matrix a = sample(4, 5, 1);
  const Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(5, 1.0, 5.0);
  const auto got = (*plan)(a, v);
  BOOST_REQUIRE_MESSAGE(got.has_value(), es::message(got.error().code));
  const Eigen::VectorXd want = a * v;
  BOOST_CHECK_EQUAL(got->rows(), 4);
  BOOST_CHECK_EQUAL(got->cols(), 1);
  BOOST_CHECK_LT((got->col(0) - want).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtMixedRank_DotProduct) {
  const auto plan = es::einsum("i,i->");
  const Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(6, 1.0, 6.0);
  const Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(6, 2.0, 7.0);
  const auto got = (*plan)(u, v);
  BOOST_REQUIRE(got.has_value());
  // A rank-0 result is 1x1 by Eigen's own convention.
  BOOST_CHECK_EQUAL(got->rows(), 1);
  BOOST_CHECK_EQUAL(got->cols(), 1);
  BOOST_CHECK_LT(std::abs((*got)(0, 0) - u.dot(v)), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtRankThree_NestFamily) {
  // "bij,bjk->bik" over a three-deep nest: every operand and the result rank 3.
  const std::vector<std::vector<std::vector<double>>> x{{{1, 2}, {3, 4}},
                                                        {{5, 6}, {7, 8}}};
  const std::vector<std::vector<std::vector<double>>> id{{{1, 0}, {0, 1}},
                                                         {{1, 0}, {0, 1}}};
  const auto plan = es::einsum("bij,bjk->bik");
  BOOST_REQUIRE(plan.has_value());
  const auto got = (*plan)(x, id);
  BOOST_REQUIRE_MESSAGE(got.has_value(), es::message(got.error().code));
  BOOST_REQUIRE_EQUAL(got->size(), 2U);
  for (std::size_t b = 0; b < 2; ++b) {
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 2; ++j) {
        BOOST_CHECK_LT(std::abs((*got)[b][i][j] - x[b][i][j]), 1e-12);
      }
    }
  }
}

// --- every way a call can be wrong -------------------------------------------
BOOST_AUTO_TEST_CASE(RtErrors_EveryCode) {
  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);

  BOOST_CHECK(
      es::failed_with((*plan)(a, sample(6, 3, 2)), errc::extent_conflict));
  BOOST_CHECK(es::failed_with((*plan)(a), errc::operand_count_mismatch));
  BOOST_CHECK(es::failed_with((*plan)(a, b, b), errc::operand_count_mismatch));

  // A nest whose rows disagree is not a tensor.
  const Nest ragged{{1.0, 2.0}, {3.0}};
  const Nest square{{1.0, 2.0}, {3.0, 4.0}};
  BOOST_CHECK(es::failed_with((*plan)(ragged, square), errc::extent_conflict));

  // A result of rank 3 has no Eigen matrix to come back as.
  const auto outer = es::einsum("ij,kl->ijkl");
  BOOST_REQUIRE(outer.has_value());
  BOOST_CHECK(es::failed_with((*outer)(a, b), errc::rank_mismatch));

  BOOST_CHECK(es::failed_with(es::einsum("abcdefghi"), errc::rank_too_high));
  BOOST_CHECK(es::failed_with(es::einsum("ij,,jk"), errc::empty_operand));
}

// --- the lowerings that are not one GEMM -------------------------------------
BOOST_AUTO_TEST_CASE(RtUnary_Transpose) {
  const auto plan = es::einsum("ij->ji");
  const Matrix a = sample(2, 3, 1);
  const auto got = (*plan)(a);
  BOOST_REQUIRE(got.has_value());
  BOOST_CHECK_LT((*got - a.transpose()).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtUnary_TraceAndDiagonal) {
  const Matrix a = sample(3, 3, 4);
  const auto trace = (*es::einsum("ii->"))(a);
  BOOST_REQUIRE(trace.has_value());
  BOOST_CHECK_LT(std::abs((*trace)(0, 0) - a.trace()), 1e-12);

  const auto diag = (*es::einsum("ii->i"))(a);
  BOOST_REQUIRE(diag.has_value());
  for (index_t i = 0; i < 3; ++i) {
    BOOST_CHECK_LT(std::abs((*diag)(i, 0) - a(i, i)), 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(RtChain_ThreeOperands) {
  const auto plan = es::einsum("ij,jk,kl->il");
  BOOST_REQUIRE(plan.has_value());
  const Matrix d = sample(3, 4, 1);
  const Matrix e = sample(4, 5, 2);
  const Matrix f = sample(5, 2, 3);
  const auto got = (*plan)(d, e, f);
  BOOST_REQUIRE(got.has_value());
  BOOST_CHECK_LT((*got - d * e * f).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtImplicitOutputIsSorted) {
  const Matrix a = sample(2, 3, 1);
  const auto got = (*es::einsum("ba"))(a);
  BOOST_REQUIRE(got.has_value());
  BOOST_CHECK_LT((*got - a.transpose()).cwiseAbs().maxCoeff(), 1e-12);
}

// --- the gather path, which the strided operands reach -----------------------
// A column-major Eigen operand and a layout_left mdspan both describe memory
// whose rows are not contiguous.  The Eigen one is still addressable where it
// lives; the layout_left mdspan is a view whose strides Eigen cannot map for
// this contraction, so it is walked through its accessor and packed.
BOOST_AUTO_TEST_CASE(RtGather_ColumnMajorEigenOperand) {
  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Matrix a = sample(4, 5, 3); // Eigen::MatrixXd is column-major
  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      b = sample(5, 3, 7);

  const auto mixed = (*plan)(a, b);
  BOOST_REQUIRE_MESSAGE(mixed.has_value(), es::message(mixed.error().code));
  BOOST_CHECK_LT((*mixed - a * b).cwiseAbs().maxCoeff(), 1e-12);

  // And with the orders swapped, so neither is the only one that works.
  const Matrix c =
      sample(3, 2, 9); // column-major again, on the right this time
  const auto flipped = (*plan)(b, c);
  BOOST_REQUIRE_MESSAGE(flipped.has_value(), es::message(flipped.error().code));
  BOOST_CHECK_LT((*flipped - b * c).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtGather_LayoutLeftMdspanOperand) {
  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);

  // Column-major storage read through layout_left: the inner stride of a row is
  // not 1, which is the case the packing path exists for.
  std::vector<double> ba(static_cast<std::size_t>(a.size()));
  std::vector<double> bb(static_cast<std::size_t>(b.size()));
  for (index_t r = 0; r < a.rows(); ++r) {
    for (index_t c = 0; c < a.cols(); ++c) {
      ba[static_cast<std::size_t>(c * a.rows() + r)] = a(r, c);
    }
  }
  for (index_t r = 0; r < b.rows(); ++r) {
    for (index_t c = 0; c < b.cols(); ++c) {
      bb[static_cast<std::size_t>(c * b.rows() + r)] = b(r, c);
    }
  }
  const std::mdspan<const double, std::dextents<std::size_t, 2>,
                    std::layout_left>
      ma{ba.data(), 4, 5};
  const std::mdspan<const double, std::dextents<std::size_t, 2>,
                    std::layout_left>
      mb{bb.data(), 5, 3};

  const auto got = (*plan)(ma, mb);
  BOOST_REQUIRE_MESSAGE(got.has_value(), es::message(got.error().code));
  BOOST_CHECK_LT((MdspanFamily::read(*got) - a * b).cwiseAbs().maxCoeff(), 1e-12);
}

// --- the Tensor family
// -------------------------------------------------------- Eigen's unsupported
// Tensor is recognised structurally, so a rank-3 result has somewhere to live
// that an Eigen matrix could not provide.
BOOST_AUTO_TEST_CASE(RtTensor_BatchedMatmul) {
  using T3 = Eigen::Tensor<double, 3, Eigen::RowMajor>;
  static_assert(es::CTensorFamily<T3> && es::rank_v<T3> == 3);
  static_assert(std::same_as<es::result_of_t<T3, T3>, T3>);

  T3 x(2, 2, 2);
  T3 id(2, 2, 2);
  x.setZero();
  id.setZero();
  double v = 1.0;
  for (int bi = 0; bi < 2; ++bi) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        x(bi, i, j) = v++;
      }
    }
    id(bi, 0, 0) = 1.0;
    id(bi, 1, 1) = 1.0;
  }

  const auto plan = es::einsum("bij,bjk->bik");
  BOOST_REQUIRE(plan.has_value());
  const auto got = (*plan)(x, id);
  BOOST_REQUIRE_MESSAGE(got.has_value(), es::message(got.error().code));
  for (int bi = 0; bi < 2; ++bi) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        BOOST_CHECK_LT(std::abs((*got)(bi, i, j) - x(bi, i, j)), 1e-12);
      }
    }
  }

  // A Tensor and a Matrix are different families and cannot share a call.
  static_assert(!es::CSameFamily<T3, Eigen::MatrixXd>);
}

// --- the geometry cache -------------------------------------------------------
// The lowering depends on the operands' shapes and strides and nothing else, so
// it is cached between calls.  What matters is that the key is complete: two
// calls of the same shape must agree, and a call whose strides differ must not
// be given the previous call's geometry.
BOOST_AUTO_TEST_CASE(RtCache_SameShapeTwiceAgrees) {
  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Matrix a = sample(6, 5, 1);
  const Matrix b = sample(5, 4, 2);
  const auto first = (*plan)(a, b);
  const auto second = (*plan)(a, b);
  BOOST_REQUIRE(first.has_value() && second.has_value());
  BOOST_CHECK_LT((*first - a * b).cwiseAbs().maxCoeff(), 1e-12);
  BOOST_CHECK_LT((*first - *second).cwiseAbs().maxCoeff(), 1e-15);
}

BOOST_AUTO_TEST_CASE(RtCache_KeyIncludesStridesAndShape) {
  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Matrix a = sample(4, 4, 3); // column-major
  const Matrix b = sample(4, 4, 5);
  const auto warm = (*plan)(a, b);
  BOOST_REQUIRE(warm.has_value());

  // Same extents, every stride swapped: a row-major operand must not be given
  // the column-major one's geometry.
  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ar = a;
  const auto swapped = (*plan)(ar, b);
  BOOST_REQUIRE_MESSAGE(swapped.has_value(), es::message(swapped.error().code));
  BOOST_CHECK_LT((*swapped - a * b).cwiseAbs().maxCoeff(), 1e-12);

  // And a different shape after that, so the cache is replaced rather than kept.
  const Matrix c = sample(3, 7, 1);
  const Matrix d = sample(7, 2, 4);
  const auto other = (*plan)(c, d);
  BOOST_REQUIRE(other.has_value());
  BOOST_CHECK_LT((*other - c * d).cwiseAbs().maxCoeff(), 1e-12);

  // Back to the first shape: still right, whatever order the calls came in.
  const auto again = (*plan)(a, b);
  BOOST_REQUIRE(again.has_value());
  BOOST_CHECK_LT((*again - a * b).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtCache_ACopyStartsCold) {
  const auto plan = es::einsum("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Matrix a = sample(4, 4, 1);
  const Matrix b = sample(4, 4, 2);
  BOOST_REQUIRE((*plan)(a, b).has_value());

  const es::Einsum copy = *plan; // caches are not copied
  const auto got = copy(a, b);
  BOOST_REQUIRE(got.has_value());
  BOOST_CHECK_LT((*got - a * b).cwiseAbs().maxCoeff(), 1e-12);
}

// The object is immutable and both call operators are const, so a const one is
// fully usable and repeated calls are independent.
BOOST_AUTO_TEST_CASE(RtObject_IsConstCallableAndRepeatable) {
  const auto plan = es::einsum("ij,jk->ik");
  const es::Einsum &e = *plan;
  const Matrix a = sample(4, 4, 1);
  const Matrix b = sample(4, 4, 2);
  const auto first = e(a, b);
  const auto second = e(a, b);
  BOOST_REQUIRE(first.has_value() && second.has_value());
  BOOST_CHECK_LT((*first - *second).cwiseAbs().maxCoeff(), 1e-15);
  BOOST_CHECK_EQUAL(e.operand_count(), 2U);
}
