// The runtime path: einsum(subscript) meeting operands whose shapes nothing
// knew until now.  The matmul matrix below is generated -- one case per operand
// family per size -- because the interesting differences between the families
// are how their memory is reached, and a hand-written case per pair would be
// three copies of one loop.
#include <unsupported/Eigen/CXX11/Tensor>

#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <cmath>
#include <string_view>
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

// The reason a call failed, or "ok" when it did not.  BOOST_REQUIRE_MESSAGE
// evaluates its message argument whether or not the check passes, and
// `.error()` on an expected that holds a value is undefined -- with assertions
// on it aborts the whole test binary, which is a far worse report than the
// failure it was meant to describe.
template <typename R>
[[nodiscard]] std::string_view why(const R &r) noexcept {
  return r.has_value() ? std::string_view{"ok"} : es::message(r.error().code);
}

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

// Row-major storage described by an mdspan: the view family, so the result
// comes back as an mdarray -- something that owns its elements, since a view
// cannot -- rather than as a view of anything.
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
  // The view family's result owns its storage: an mdarray, contiguous and
  // layout_right, rather than the nest the other non-Eigen family answers.
  template <typename A> [[nodiscard]] static Matrix read(const A &r) {
    Matrix out(static_cast<index_t>(r.extent(0)),
               static_cast<index_t>(r.extent(1)));
    for (std::size_t i = 0; i < r.extent(0); ++i) {
      for (std::size_t j = 0; j < r.extent(1); ++j) {
        out(static_cast<index_t>(i), static_cast<index_t>(j)) = r[i, j];
      }
    }
    return out;
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
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
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
// Every family's result type in one place.  Each of these is a rule from the
// spec rather than an observation about the code, and each has been wrong at
// least once: the view family's mdarray was reintroduced as a nest by an edit
// meant for another branch, and the assertion that should have caught it had
// been rewritten to agree.  A single table is harder to quietly amend than a
// line buried in a behavioural test.
BOOST_AUTO_TEST_CASE(RtResult_TypeTable) {
  // Eigen in, an Eigen matrix out, in the operands' own storage order.
  static_assert(std::same_as<es::result_of_t<Matrix, Matrix>, Matrix>);
  static_assert(std::same_as<es::result_of_t<Eigen::MatrixXf, Eigen::MatrixXf>,
                             Eigen::MatrixXf>);

  // A nest deep enough for the result answers its own type, so an einsum over
  // std::vector nests gives back a std::vector nest.
  static_assert(std::same_as<es::result_of_t<Nest, Nest>, Nest>);
  using Nest3 = std::vector<std::vector<std::vector<double>>>;
  static_assert(std::same_as<es::result_of_t<Nest3, Nest3>, Nest3>);

  // A view owns nothing, so it answers the owning member of its own family --
  // an mdarray, dynamically sized because the extents are not in the type, and
  // layout_right because the executor writes into it contiguously.
  using Md2 = std::mdspan<const double, std::dextents<std::size_t, 2>>;
  using Md3 = std::mdspan<const double, std::dextents<std::size_t, 3>>;
  static_assert(
      std::same_as<es::result_of_t<Md2, Md2>,
                   std::experimental::mdarray<double, std::dextents<std::size_t, 2>,
                                              std::layout_right>>);
  // The rank is the widest operand's, not the first's.
  static_assert(
      std::same_as<es::result_of_t<Md2, Md3>,
                   std::experimental::mdarray<double, std::dextents<std::size_t, 3>,
                                              std::layout_right>>);

  // A Tensor is its own family: a rank-3 result cannot be a Matrix, and the
  // type cannot be rebuilt here, so the widest operand's type is the answer.
  using T3 = Eigen::Tensor<double, 3, Eigen::RowMajor>;
  using T2 = Eigen::Tensor<double, 2, Eigen::RowMajor>;
  static_assert(std::same_as<es::result_of_t<T3, T3>, T3>);
  static_assert(std::same_as<es::result_of_t<T2, T3>, T3>);

  // And the compile-time path, where a static-extent view's extents are known:
  // the same mdarray, but over a std::array, so the call allocates nothing.
  const auto fixed = es::einsum<"ij,jk->ik">();
  std::array<double, 9> xs{};
  const std::mdspan<const double, std::extents<std::size_t, 3, 3>> x{xs.data()};
  const auto got = fixed(x, x);
  static_assert(
      std::same_as<std::remove_cvref_t<decltype(*got)>,
                   std::experimental::mdarray<double, std::extents<std::size_t, 3, 3>,
                                              std::layout_right,
                                              std::array<double, 9>>>);
  BOOST_CHECK(got.has_value());
}

BOOST_AUTO_TEST_CASE(RtResult_FamilyDecidesTheType) {
  const auto plan = es::einsum("ij,jk->ik");
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);
  const auto eigen = (*plan)(a, b);
  static_assert(std::same_as<std::remove_cvref_t<decltype(*eigen)>, Matrix>);
  BOOST_REQUIRE(eigen.has_value());
  BOOST_CHECK_LT((*eigen - a * b).cwiseAbs().maxCoeff(), 1e-12);

  // A view cannot own a result, so the view family answers the owning member
  // of its own family -- a dynamic-extent mdarray, not a nest of vectors.
  using Owned = std::experimental::mdarray<double, std::dextents<std::size_t, 2>,
                                           std::layout_right>;
  const MdspanFamily left{a};
  const MdspanFamily right{b};
  const auto viewed = (*plan)(left.operand(), right.operand());
  static_assert(std::same_as<std::remove_cvref_t<decltype(*viewed)>, Owned>);
  BOOST_REQUIRE(viewed.has_value());
  BOOST_CHECK_LT((MdspanFamily::read(*viewed) - a * b).cwiseAbs().maxCoeff(),
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
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
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
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
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
  BOOST_REQUIRE_MESSAGE(mixed.has_value(), why(mixed));
  BOOST_CHECK_LT((*mixed - a * b).cwiseAbs().maxCoeff(), 1e-12);

  // And with the orders swapped, so neither is the only one that works.
  const Matrix c =
      sample(3, 2, 9); // column-major again, on the right this time
  const auto flipped = (*plan)(b, c);
  BOOST_REQUIRE_MESSAGE(flipped.has_value(), why(flipped));
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
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
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
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
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
  BOOST_REQUIRE_MESSAGE(swapped.has_value(), why(swapped));
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

// --- ellipsis and broadcasting ------------------------------------------------
// Everything below is checked against the loop the subscript describes, because
// a broadcast that silently read the wrong element would still produce numbers.

BOOST_AUTO_TEST_CASE(RtBroadcast_DiagonalIsSquareInItsOwnOperand) {
  // "i,ii->i" with (5) and (1, 1): the second operand's diagonal IS square, so
  // it broadcasts to 5 -- the repeat rule is about that operand's own two axes,
  // not about what the label is bound to elsewhere.
  const auto plan = es::einsum("i,ii->i");
  BOOST_REQUIRE(plan.has_value());
  Eigen::VectorXd a(5);
  a << 1, 2, 3, 4, 5;
  Eigen::MatrixXd b(1, 1);
  b << 2.0;
  const auto got = (*plan)(a, b);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
  BOOST_REQUIRE_EQUAL(got->rows(), 5);
  for (index_t i = 0; i < 5; ++i) {
    BOOST_CHECK_LT(std::abs((*got)(i, 0) - a(i) * 2.0), 1e-12);
  }

  // A diagonal that is not square is still an error, whatever broadcasting the
  // label does elsewhere.
  Eigen::MatrixXd oblong(1, 3);
  oblong.setOnes();
  BOOST_CHECK(es::failed_with((*plan)(a, oblong), errc::extent_conflict));
}

BOOST_AUTO_TEST_CASE(RtEllipsis_BatchedMatmulOverTensors) {
  using T3 = Eigen::Tensor<double, 3, Eigen::RowMajor>;
  const auto plan = es::einsum("...ij,...jk->...ik");
  BOOST_REQUIRE(plan.has_value());
  T3 x(2, 2, 3);
  T3 y(2, 3, 2);
  double v = 1.0;
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        x(b, i, j) = v++;
      }
    }
  }
  v = 1.0;
  for (int b = 0; b < 2; ++b) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 2; ++k) {
        y(b, j, k) = v++;
      }
    }
  }
  const auto got = (*plan)(x, y);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 2; ++i) {
      for (int k = 0; k < 2; ++k) {
        double want = 0.0;
        for (int j = 0; j < 3; ++j) {
          want += x(b, i, j) * y(b, j, k);
        }
        BOOST_CHECK_LT(std::abs((*got)(b, i, k) - want), 1e-12);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(RtEllipsis_LeftOnlyAndRightAligned) {
  // The left operand's '...' covers one axis and the right one has none, so the
  // broadcast axes are right-aligned: (5, 2, 3) meets (3, 4) on their trailing
  // axes and the 5 rides along.
  using T3 = Eigen::Tensor<double, 3, Eigen::RowMajor>;
  const auto plan = es::einsum("...ij,jk->...ik");
  BOOST_REQUIRE(plan.has_value());
  T3 x(5, 2, 3);
  Eigen::Tensor<double, 2, Eigen::RowMajor> y(3, 4);
  double v = 1.0;
  for (int b = 0; b < 5; ++b) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        x(b, i, j) = v++;
      }
    }
  }
  v = 0.5;
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 4; ++k) {
      y(j, k) = v++;
    }
  }
  const auto got = (*plan)(x, y);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
  for (int b = 0; b < 5; ++b) {
    for (int i = 0; i < 2; ++i) {
      for (int k = 0; k < 4; ++k) {
        double want = 0.0;
        for (int j = 0; j < 3; ++j) {
          want += x(b, i, j) * y(j, k);
        }
        BOOST_CHECK_LT(std::abs((*got)(b, i, k) - want), 1e-12);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(RtBroadcast_NamedLabelStretches) {
  // A named label of extent 1 against one of extent 4: the operand is read at
  // one offset for every index along that axis.
  const auto plan = es::einsum("ij,ij->ij");
  BOOST_REQUIRE(plan.has_value());
  const Nest thin{{2.0}, {3.0}, {4.0}};                 // (3, 1)
  const Nest wide{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  const auto got = (*plan)(thin, wide);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
  BOOST_REQUIRE_EQUAL(got->size(), 3U);
  BOOST_REQUIRE_EQUAL((*got)[0].size(), 4U);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      BOOST_CHECK_LT(std::abs((*got)[i][j] - thin[i][0] * wide[i][j]), 1e-12);
    }
  }

  // Neither equal nor 1 is still a conflict.
  const Nest other{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  BOOST_CHECK(es::failed_with((*plan)(other, wide), errc::extent_conflict));
}

BOOST_AUTO_TEST_CASE(RtBroadcast_InsideAReducedLabel) {
  // The same stretch, but the axis is summed away rather than kept.
  const auto plan = es::einsum("ij,ij->");
  BOOST_REQUIRE(plan.has_value());
  const Nest thin{{2.0}, {3.0}, {4.0}};
  const Nest wide{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  const auto got = (*plan)(thin, wide);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
  double want = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      want += thin[i][0] * wide[i][j];
    }
  }
  BOOST_CHECK_LT(std::abs((*got)[0][0] - want), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtEllipsis_DiagonalAfterExpansion) {
  // "...ii->...i": the expansion runs first, and what is left is an ordinary
  // batched diagonal.
  using T3 = Eigen::Tensor<double, 3, Eigen::RowMajor>;
  const auto plan = es::einsum("...ii->...i");
  BOOST_REQUIRE(plan.has_value());
  T3 x(2, 3, 3);
  double v = 1.0;
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        x(b, i, j) = v++;
      }
    }
  }
  const auto got = (*plan)(x);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));

  // The contraction is rank 2, but on this path the result TYPE is the widest
  // operand's -- a rank-3 Tensor -- because the return type cannot see the
  // subscript.  A lower output rank is padded with leading extents of 1, so the
  // answer arrives as (1, 2, 3) and is read with that leading index.
  BOOST_REQUIRE_EQUAL(got->dimension(0), 1);
  BOOST_REQUIRE_EQUAL(got->dimension(1), 2);
  BOOST_REQUIRE_EQUAL(got->dimension(2), 3);
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 3; ++i) {
      BOOST_CHECK_LT(std::abs((*got)(0, b, i) - x(b, i, i)), 1e-12);
    }
  }
}

BOOST_AUTO_TEST_CASE(RtEllipsis_OutputMustNameTheBroadcastAxes) {
  // An explicit output with no '...' while the operands' one covers an axis.
  const auto plan = es::einsum("...ij,...jk->ik");
  BOOST_REQUIRE(plan.has_value());
  using T3 = Eigen::Tensor<double, 3, Eigen::RowMajor>;
  T3 x(2, 2, 2);
  T3 y(2, 2, 2);
  x.setConstant(1.0);
  y.setConstant(1.0);
  BOOST_CHECK(es::failed_with((*plan)(x, y), errc::ellipsis_not_in_output));

  // With nothing for it to cover, the same subscript shape is fine.
  const auto flat = es::einsum("...i->i");
  BOOST_REQUIRE(flat.has_value());
  const Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(4, 1.0, 4.0);
  const auto ok = (*flat)(v);
  BOOST_REQUIRE_MESSAGE(ok.has_value(), why(ok));
  BOOST_CHECK_LT((ok->col(0) - v).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtBroadcast_MdspanEllipsisStretchesTheBatch) {
  // The view family under '...': a (1, 2, 3) against a (5, 3, 4).  The left
  // operand's broadcast axis is 1, so it is read at one offset for all five
  // batches, and the result is an mdarray of (5, 2, 4).
  using Md3 = std::mdspan<const double, std::dextents<std::size_t, 3>>;
  const auto plan = es::einsum("...ij,...jk->...ik");
  BOOST_REQUIRE(plan.has_value());

  std::vector<double> xs(1 * 2 * 3);
  std::vector<double> ys(5 * 3 * 4);
  for (std::size_t n = 0; n < xs.size(); ++n) {
    xs[n] = static_cast<double>(n) + 1.0;
  }
  for (std::size_t n = 0; n < ys.size(); ++n) {
    ys[n] = static_cast<double>((n * 7) % 11) - 5.0;
  }
  const Md3 x{xs.data(), 1, 2, 3};
  const Md3 y{ys.data(), 5, 3, 4};

  const auto got = (*plan)(x, y);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
  BOOST_REQUIRE_EQUAL(got->extent(0), 5U);
  BOOST_REQUIRE_EQUAL(got->extent(1), 2U);
  BOOST_REQUIRE_EQUAL(got->extent(2), 4U);
  for (std::size_t b = 0; b < 5; ++b) {
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t k = 0; k < 4; ++k) {
        double want = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
          want += (x[0, i, j]) * (y[b, j, k]);
        }
        BOOST_CHECK_LT(std::abs(((*got)[b, i, k]) - want), 1e-12);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(RtBroadcast_StrideZeroOnTheGemmBatchAxis) {
  // "bij,bjk->bik" with b = 1 on the left and b = 5 on the right.  The stretched
  // axis is the GEMM's batch, so the left slab is addressed at the same offset
  // for every batch -- a different path from the hadamard case above, where the
  // stretched axis is one the kernel iterates elementwise.
  const auto plan = es::einsum("bij,bjk->bik");
  BOOST_REQUIRE(plan.has_value());

  std::vector<std::vector<std::vector<double>>> l(
      1, std::vector<std::vector<double>>(2, std::vector<double>(3, 0.0)));
  double v = 1.0;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      l[0][i][j] = v++;
    }
  }
  std::vector<std::vector<std::vector<double>>> r(
      5, std::vector<std::vector<double>>(3, std::vector<double>(4, 0.0)));
  v = 0.5;
  for (std::size_t b = 0; b < 5; ++b) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        r[b][j][k] = v++;
      }
    }
  }

  const auto got = (*plan)(l, r);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
  BOOST_REQUIRE_EQUAL(got->size(), 5U);
  for (std::size_t b = 0; b < 5; ++b) {
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t k = 0; k < 4; ++k) {
        double want = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
          want += l[0][i][j] * r[b][j][k];
        }
        BOOST_CHECK_LT(std::abs((*got)[b][i][k] - want), 1e-12);
      }
    }
  }
}

// --- the contraction path -----------------------------------------------------
namespace {

[[nodiscard]] es::BoundExtents extents_of(
    const std::initializer_list<std::pair<char, index_t>> pairs) {
  es::BoundExtents bound;
  for (const auto &[c, e] : pairs) {
    bound[c] = {.extent = e, .known = true, .broadcast = false};
  }
  return bound;
}

} // namespace

BOOST_AUTO_TEST_CASE(RtPath_GreedyContractsTheCheapPairFirst) {
  // A thin middle: (i j)(j k) is a 2x2, so it goes first.
  const auto thin = es::einsum("ij,jk,kl->il");
  BOOST_REQUIRE(thin.has_value());
  const Matrix a = sample(2, 100, 1);
  const Matrix b = sample(100, 2, 2);
  const Matrix c = sample(2, 100, 3);
  const auto got = (*thin)(a, b, c);
  BOOST_REQUIRE_MESSAGE(got.has_value(), why(got));
  BOOST_CHECK_LT((*got - a * b * c).cwiseAbs().maxCoeff(), 1e-9);

  const es::Path &chosen = es::impl::einsum_access::last_path(*thin);
  BOOST_REQUIRE_EQUAL(chosen.steps.size(), 2U);
  BOOST_CHECK_EQUAL(int{chosen.steps[0].l_src}, 0);
  BOOST_CHECK_EQUAL(int{chosen.steps[0].r_src}, 1);

  // A fat middle: now the LAST pair is the cheap one.
  const auto fat = es::einsum("ab,bc,cd->ad");
  BOOST_REQUIRE(fat.has_value());
  const Matrix d = sample(100, 2, 1);
  const Matrix e = sample(2, 100, 2);
  const Matrix f = sample(100, 2, 3);
  const auto other = (*fat)(d, e, f);
  BOOST_REQUIRE_MESSAGE(other.has_value(), why(other));
  BOOST_CHECK_LT((*other - d * e * f).cwiseAbs().maxCoeff(), 1e-9);

  const es::Path &second = es::impl::einsum_access::last_path(*fat);
  BOOST_REQUIRE_EQUAL(second.steps.size(), 2U);
  BOOST_CHECK_EQUAL(int{second.steps[0].l_src}, 1);
  BOOST_CHECK_EQUAL(int{second.steps[0].r_src}, 2);
}

BOOST_AUTO_TEST_CASE(RtPath_SequentialIsTheSubscriptsOwnOrder) {
  const auto plan = es::einsum("ab,bc,cd->ad", es::path::sequential);
  BOOST_REQUIRE(plan.has_value());
  BOOST_CHECK(es::impl::einsum_access::order_of(*plan) == es::path::sequential);

  const Matrix d = sample(100, 2, 1);
  const Matrix e = sample(2, 100, 2);
  const Matrix f = sample(100, 2, 3);
  const auto got = (*plan)(d, e, f);
  BOOST_REQUIRE(got.has_value());
  BOOST_CHECK_LT((*got - d * e * f).cwiseAbs().maxCoeff(), 1e-9);

  // Left to right, verbatim: operands 0 and 1, then that result with operand 2.
  const es::Path &chosen = es::impl::einsum_access::last_path(*plan);
  BOOST_REQUIRE_EQUAL(chosen.steps.size(), 2U);
  BOOST_CHECK_EQUAL(int{chosen.steps[0].l_src}, 0);
  BOOST_CHECK_EQUAL(int{chosen.steps[0].r_src}, 1);
  BOOST_CHECK_EQUAL(int{chosen.steps[1].l_src}, int{es::kIntermediate});
  BOOST_CHECK_EQUAL(int{chosen.steps[1].r_src}, 2);
  BOOST_CHECK(chosen.steps[1].writes_output);
}

BOOST_AUTO_TEST_CASE(RtPath_GreedyIsNeverDearerThanSequential) {
  const Matrix d = sample(100, 2, 1);
  const Matrix e = sample(2, 100, 2);
  const Matrix f = sample(100, 2, 3);
  const auto bound = extents_of({{'a', 100}, {'b', 2}, {'c', 100}, {'d', 2}});

  const auto greedy = es::einsum("ab,bc,cd->ad", es::path::greedy);
  const auto step_by_step = es::einsum("ab,bc,cd->ad", es::path::sequential);
  BOOST_REQUIRE(greedy.has_value() && step_by_step.has_value());
  const auto one = (*greedy)(d, e, f);
  const auto two = (*step_by_step)(d, e, f);
  BOOST_REQUIRE(one.has_value() && two.has_value());

  // The same answer either way, and the chosen order costs no more.
  BOOST_CHECK_LT((*one - *two).cwiseAbs().maxCoeff(), 1e-9);
  const auto cheap = es::impl::flops(es::impl::einsum_access::last_path(*greedy), bound);
  const auto plain = es::impl::flops(es::impl::einsum_access::last_path(*step_by_step), bound);
  BOOST_CHECK_LE(cheap, plain);
  BOOST_CHECK_LT(cheap, plain); // on this shape it is strictly better
}
