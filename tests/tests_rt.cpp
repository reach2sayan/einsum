// The runtime path: rt::plan(subscript) meeting operands whose shapes nothing
// knew until now.  The matmul matrix below is generated -- one case per operand
// kind per size -- because the interesting differences between the kinds are
// their strides, and a hand-written case per pair would be seven copies of one
// loop.
#include "einsum/einsum.hpp"
#include "einsum/rt/parse.hpp"

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include <cstdlib>
#include <string_view>
#include <new>
#include <vector>

#define BOOST_TEST_MODULE EinsumRuntimeSuite
#include <boost/test/included/unit_test.hpp>

namespace es = einsum;
using es::errc;
using es::index_t;

namespace {

using Matrix = Eigen::MatrixXd;

[[nodiscard]] Matrix sample(const index_t rows, const index_t cols, const int seed) {
  Matrix m(rows, cols);
  for (index_t r = 0; r < rows; ++r) {
    for (index_t c = 0; c < cols; ++c) {
      m(r, c) = static_cast<double>((r * 7 + c * 3 + seed) % 11) - 5.0;
    }
  }
  return m;
}

// --- one holder per rung of the operand ladder -------------------------------
// Each fills itself from a matrix, hands back something bind() takes, offers a
// place to write into, and reads back as a matrix.  Nothing else about them is
// alike, which is the point.
template <index_t N> struct EigenRowMajor {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m{N, N};
  explicit EigenRowMajor(const Matrix &src) { m = src; }
  EigenRowMajor() { m.setZero(); }
  [[nodiscard]] const auto &operand() const { return m; }
  [[nodiscard]] auto output() { return es::into(m); }
  [[nodiscard]] Matrix read() const { return m; }
};

template <index_t N> struct EigenColMajor {
  Matrix m{N, N};
  explicit EigenColMajor(const Matrix &src) : m{src} {}
  EigenColMajor() { m.setZero(); }
  [[nodiscard]] const auto &operand() const { return m; }
  [[nodiscard]] auto output() { return es::into(m); }
  [[nodiscard]] Matrix read() const { return m; }
};

// Row-major storage shared by the four holders below; only how they describe it
// differs.
template <index_t N> struct RowMajorBuffer {
  std::vector<double> buffer = std::vector<double>(static_cast<std::size_t>(N * N), 0.0);
  RowMajorBuffer() = default;
  explicit RowMajorBuffer(const Matrix &src) {
    for (index_t r = 0; r < N; ++r) {
      for (index_t c = 0; c < N; ++c) {
        buffer[static_cast<std::size_t>(r * N + c)] = src(r, c);
      }
    }
  }
  [[nodiscard]] Matrix read() const {
    Matrix m(N, N);
    for (index_t r = 0; r < N; ++r) {
      for (index_t c = 0; c < N; ++c) {
        m(r, c) = buffer[static_cast<std::size_t>(r * N + c)];
      }
    }
    return m;
  }
};

template <index_t N> struct MdspanStatic : RowMajorBuffer<N> {
  using RowMajorBuffer<N>::RowMajorBuffer;
  using extents_type = std::extents<std::size_t, static_cast<std::size_t>(N),
                                    static_cast<std::size_t>(N)>;
  [[nodiscard]] auto operand() const {
    return std::mdspan<const double, extents_type>{this->buffer.data()};
  }
  [[nodiscard]] auto output() {
    return es::into(std::mdspan<double, extents_type>{this->buffer.data()});
  }
};

template <index_t N> struct MdspanDynamic : RowMajorBuffer<N> {
  using RowMajorBuffer<N>::RowMajorBuffer;
  using extents_type = std::dextents<std::size_t, 2>;
  [[nodiscard]] auto operand() const {
    return std::mdspan<const double, extents_type>{this->buffer.data(),
                                                   static_cast<std::size_t>(N),
                                                   static_cast<std::size_t>(N)};
  }
  [[nodiscard]] auto output() {
    return es::into(std::mdspan<double, extents_type>{
        this->buffer.data(), static_cast<std::size_t>(N), static_cast<std::size_t>(N)});
  }
};

// Column-major storage read through layout_left: the case where the inner
// stride of a row is not 1 and the slab has to be mapped the other way round.
template <index_t N> struct MdspanLeft {
  std::vector<double> buffer = std::vector<double>(static_cast<std::size_t>(N * N), 0.0);
  MdspanLeft() = default;
  explicit MdspanLeft(const Matrix &src) {
    for (index_t r = 0; r < N; ++r) {
      for (index_t c = 0; c < N; ++c) {
        buffer[static_cast<std::size_t>(c * N + r)] = src(r, c);
      }
    }
  }
  using extents_type = std::dextents<std::size_t, 2>;
  [[nodiscard]] auto operand() const {
    return std::mdspan<const double, extents_type, std::layout_left>{
        buffer.data(), static_cast<std::size_t>(N), static_cast<std::size_t>(N)};
  }
  [[nodiscard]] auto output() {
    return es::into(std::mdspan<double, extents_type, std::layout_left>{
        buffer.data(), static_cast<std::size_t>(N), static_cast<std::size_t>(N)});
  }
  [[nodiscard]] Matrix read() const {
    Matrix m(N, N);
    for (index_t r = 0; r < N; ++r) {
      for (index_t c = 0; c < N; ++c) {
        m(r, c) = buffer[static_cast<std::size_t>(c * N + r)];
      }
    }
    return m;
  }
};

template <index_t N> struct VectorShape : RowMajorBuffer<N> {
  using RowMajorBuffer<N>::RowMajorBuffer;
  [[nodiscard]] auto operand() const { return es::flat(this->buffer, {N, N}); }
  [[nodiscard]] auto output() { return es::into(this->buffer, {N, N}); }
};

template <index_t N> struct PointerShape : RowMajorBuffer<N> {
  using RowMajorBuffer<N>::RowMajorBuffer;
  [[nodiscard]] auto operand() const { return es::flat(this->buffer.data(), {N, N}); }
  [[nodiscard]] auto output() { return es::into(this->buffer.data(), {N, N}); }
};

// --- the generated matmul cases ----------------------------------------------
template <template <index_t> class Kind, index_t N> void check_matmul() {
  const Matrix a = sample(N, N, 1);
  const Matrix b = sample(N, N, 5);
  const Kind<N> left{a};
  const Kind<N> right{b};
  Kind<N> out;

  const auto plan = es::rt::plan("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const auto ok = (*plan)(left.operand(), right.operand(), out.output());
  if (!ok) {
    BOOST_ERROR(es::message(ok.error().code));
    return;
  }
  BOOST_CHECK_LT((out.read() - a * b).cwiseAbs().maxCoeff(), 1e-12);
}

} // namespace

#define EINSUM_KINDS                                                                    \
  (EigenRowMajor)(EigenColMajor)(MdspanStatic)(MdspanDynamic)(MdspanLeft)(VectorShape)  \
      (PointerShape)
#define EINSUM_SIZES (2)(3)(4)(8)(16)(33)

#define EINSUM_MATMUL_CASE(r, product)                                                  \
  BOOST_AUTO_TEST_CASE(BOOST_PP_CAT(BOOST_PP_CAT(RtMatmul_, BOOST_PP_SEQ_ELEM(0, product)), \
                                    BOOST_PP_CAT(_, BOOST_PP_SEQ_ELEM(1, product)))) {  \
    check_matmul<BOOST_PP_SEQ_ELEM(0, product), BOOST_PP_SEQ_ELEM(1, product)>();       \
  }
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EINSUM_MATMUL_CASE, (EINSUM_KINDS)(EINSUM_SIZES))

// --- the kind of the result --------------------------------------------------
BOOST_AUTO_TEST_CASE(RtResult_EigenInEigenOut) {
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);
  const auto plan = es::rt::plan("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const auto got = (*plan)(a, b);
  static_assert(std::same_as<std::remove_cvref_t<decltype(got)>, es::result<Matrix>>);
  BOOST_REQUIRE(got.has_value());
  BOOST_CHECK_LT((*got - a * b).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtResult_MdspanInMdspanOut) {
  MdspanStatic<4> left{sample(4, 4, 3)};
  MdspanStatic<4> right{sample(4, 4, 9)};
  const auto plan = es::rt::plan("ij,jk->ik");
  const auto got = (*plan)(left.operand(), right.operand());
  BOOST_REQUIRE(got.has_value());
  const auto view = got->as_mdspan<2>();
  BOOST_REQUIRE(view.has_value());
  BOOST_CHECK_EQUAL(view->extent(0), 4U);
  BOOST_CHECK_EQUAL(view->extent(1), 4U);
  const Matrix want = left.read() * right.read();
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) {
      BOOST_CHECK_LT(std::abs((*view)[r, c] - want(static_cast<index_t>(r),
                                                   static_cast<index_t>(c))),
                     1e-12);
    }
  }
  // The rank is not in the plan's type, so asking for the wrong one is an error
  // rather than a compile failure.
  BOOST_CHECK(es::failed_with(got->as_mdspan<3>(), errc::rank_mismatch));
}

BOOST_AUTO_TEST_CASE(RtResult_FlatInVectorOut) {
  const std::vector<double> a(20, 1.0);
  const std::vector<double> b(15, 2.0);
  const auto plan = es::rt::plan("ij,jk->ik");
  const auto got = (*plan)(es::flat(a, {4, 5}), es::flat(b, {5, 3}));
  static_assert(std::same_as<std::remove_cvref_t<decltype(got)>, es::result<std::vector<double>>>);
  BOOST_REQUIRE(got.has_value());
  BOOST_CHECK_EQUAL(got->size(), 12U);
  for (const double x : *got) {
    BOOST_CHECK_LT(std::abs(x - 10.0), 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(RtResult_PointerIsFlatToo) {
  const std::vector<double> a(20, 1.0);
  const std::vector<double> b(15, 2.0);
  std::vector<double> c(12, 0.0);
  const auto plan = es::rt::plan("ij,jk->ik");
  const auto ok = (*plan)(es::flat(a.data(), {4, 5}), es::flat(b.data(), {5, 3}),
                          es::into(c.data(), {4, 3}));
  static_assert(std::same_as<std::remove_cvref_t<decltype(ok)>, es::result<void>>);
  BOOST_REQUIRE(ok.has_value());
  for (const double x : c) {
    BOOST_CHECK_LT(std::abs(x - 10.0), 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(RtResult_RankThreeIsNoEigenMatrix) {
  const Matrix a = sample(2, 2, 1);
  const Matrix b = sample(2, 2, 2);
  const auto plan = es::rt::plan("ij,kl->ijkl");
  BOOST_CHECK(es::failed_with((*plan)(a, b), errc::not_matrix));
}

// --- to_matrix ---------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RtToMatrix_Ranks) {
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);
  BOOST_CHECK_LT(
      ((*es::rt::plan("ij,jk->ik")).to_matrix(a, b).value() - a * b).cwiseAbs().maxCoeff(),
      1e-12);

  const Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(6, 1.0, 6.0);
  const Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(6, 2.0, 7.0);
  const auto scalar = (*es::rt::plan("i,i->")).to_matrix(u, v);
  BOOST_REQUIRE(scalar.has_value());
  BOOST_CHECK_EQUAL(scalar->rows(), 1);
  BOOST_CHECK_EQUAL(scalar->cols(), 1);
  BOOST_CHECK_LT(std::abs((*scalar)(0, 0) - u.dot(v)), 1e-12);

  const Eigen::VectorXd ones = Eigen::VectorXd::Ones(5);
  const auto vector = (*es::rt::plan("ij,j->i")).to_matrix(a, ones);
  BOOST_REQUIRE(vector.has_value());
  BOOST_CHECK_EQUAL(vector->rows(), 4);
  BOOST_CHECK_EQUAL(vector->cols(), 1);

  BOOST_CHECK(
      es::failed_with((*es::rt::plan("ij,kl->ijkl")).to_matrix(a, b), errc::not_matrix));
}

// --- every way a call can be wrong -------------------------------------------
BOOST_AUTO_TEST_CASE(RtErrors_EveryCode) {
  const auto plan = es::rt::plan("ij,jk->ik");
  BOOST_REQUIRE(plan.has_value());
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);
  Matrix out(4, 3);

  BOOST_CHECK(es::failed_with(plan->bind(a, sample(6, 3, 2), es::into(out)),
                              errc::extent_conflict));
  BOOST_CHECK(es::failed_with(plan->bind(a, es::into(out)), errc::operand_count_mismatch));
  {
    Matrix wrong(9, 9);
    BOOST_CHECK(es::failed_with(plan->bind(a, b, es::into(wrong)), errc::output_mismatch));
  }
  {
    const Eigen::VectorXd flat = Eigen::VectorXd::Ones(20);
    BOOST_CHECK(es::failed_with(plan->bind(flat, b, es::into(out)), errc::rank_mismatch));
  }
  {
    const std::vector<double> small(6, 1.0);
    BOOST_CHECK(es::failed_with(plan->bind(es::flat(small, {4, 5}), es::flat(small, {5, 3}),
                                           es::into(out)),
                                errc::size_mismatch));
  }
  {
    std::vector<double> tiny;
    const auto chain = es::rt::plan("ij,jk,kl->il");
    const Matrix d = sample(3, 4, 1);
    const Matrix e = sample(4, 5, 2);
    const Matrix f = sample(5, 2, 3);
    Matrix g(3, 2);
    BOOST_CHECK(es::failed_with(
        chain->bind(d, e, f, es::into(g), std::span<double>{tiny}), errc::workspace_too_small));
  }
  BOOST_CHECK(es::failed_with(es::rt::plan("abcdefghi"), errc::rank_too_high));
}

// --- rebinding ---------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RtRebind_PointersOnly) {
  const auto plan = es::rt::plan("ij,jk->ik");
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);
  Matrix out(4, 3);
  auto bound = plan->bind(a, b, es::into(out));
  BOOST_REQUIRE(bound.has_value());
  bound->eval();
  BOOST_CHECK_LT((out - a * b).cwiseAbs().maxCoeff(), 1e-12);

  const Matrix a2 = sample(4, 5, 7);
  const Matrix b2 = sample(5, 3, 4);
  Matrix out2(4, 3);
  BOOST_REQUIRE(bound->eval(a2, b2, es::into(out2)).has_value());
  BOOST_CHECK_LT((out2 - a2 * b2).cwiseAbs().maxCoeff(), 1e-12);

  const Matrix wrong = sample(4, 6, 1);
  const Matrix wrong2 = sample(6, 3, 1);
  BOOST_CHECK(es::failed_with(bound->eval(wrong, wrong2, es::into(out2)),
                              errc::extent_conflict));
}

// Same extents, different strides.  A Geometry is a set of offsets, so an
// operand that only moved its rows is a different operand -- and rebinding,
// which swaps pointers and nothing else, has to say so.
BOOST_AUTO_TEST_CASE(RtRebind_SameExtentsDifferentStridesIsRefused) {
  const auto plan = es::rt::plan("ij,jk->ik");
  const Matrix a = sample(4, 4, 1);
  const Matrix b = sample(4, 3, 2);
  Matrix out(4, 3);
  auto bound = plan->bind(a, b, es::into(out));
  BOOST_REQUIRE(bound.has_value());
  bound->eval();

  // The same 4x4 numbers stored the other way round: identical extents, every
  // stride swapped.  (Eigen's own .eval() of a transpose lands here too -- the
  // plain object of a Transpose<ColMajor> is row-major.)
  const Matrix square = sample(4, 4, 6);
  const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major =
      square;
  BOOST_CHECK(es::failed_with(bound->eval(row_major, b, es::into(out)),
                              errc::extent_conflict));

  // The same numbers in the layout it was bound to are accepted, and the
  // refusal above left the binding untouched.
  BOOST_REQUIRE(bound->eval(square, b, es::into(out)).has_value());
  BOOST_CHECK_LT((out - square * b).cwiseAbs().maxCoeff(), 1e-12);
}

// --- the scatter path --------------------------------------------------------
// An output whose m and n axes do not nest the way the GEMM writes them cannot
// be Mapped, so the step lands in scratch and is scattered out afterwards.
// Nothing else in the suite reaches that branch.
BOOST_AUTO_TEST_CASE(RtScatter_TransposedMatrixOutput) {
  const Matrix a = sample(4, 5, 1);
  const Matrix b = sample(5, 3, 2);
  // "ij,jk->ki": the output's axes are (k, i) while the product writes (i, k).
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(3, 4);
  const auto plan = es::rt::plan("ij,jk->ki");
  BOOST_REQUIRE(plan.has_value());
  BOOST_REQUIRE((*plan)(a, b, es::into(out)).has_value());
  BOOST_CHECK_LT((out - (a * b).transpose()).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RtScatter_PermutedRank3Output) {
  // "ijk,kl->lij": the output puts l outermost, so neither the m group (i, j)
  // nor the n group (l) sits where a plain Map would want it.
  std::vector<double> a(2 * 3 * 4);
  std::vector<double> b(4 * 5);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<double>((i * 3) % 7) - 3.0;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<double>((i * 5) % 9) - 4.0;
  }
  std::vector<double> out(5 * 2 * 3, 0.0);
  std::mdspan<const double, std::extents<std::size_t, 2, 3, 4>> mda{a.data()};
  std::mdspan<const double, std::extents<std::size_t, 4, 5>> mdb{b.data()};
  std::mdspan<double, std::extents<std::size_t, 5, 2, 3>> mdo{out.data()};

  const auto plan = es::rt::plan("ijk,kl->lij");
  BOOST_REQUIRE(plan.has_value());
  BOOST_REQUIRE((*plan)(mda, mdb, es::into(mdo)).has_value());

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t l = 0; l < 5; ++l) {
        double want = 0.0;
        for (std::size_t k = 0; k < 4; ++k) {
          want += (mda[i, j, k]) * (mdb[k, l]);
        }
        BOOST_CHECK_LT(std::abs((mdo[l, i, j]) - want), 1e-12);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(RtScatter_ColumnMajorEigenResultFromTheOwningForm) {
  const Matrix a = sample(6, 4, 3);
  const Matrix b = sample(4, 7, 8);
  const auto got = (*es::rt::plan("ij,jk->ik"))(a, b);
  BOOST_REQUIRE(got.has_value());
  const Matrix want = a * b;
  BOOST_REQUIRE_EQUAL(got->rows(), want.rows());
  BOOST_REQUIRE_EQUAL(got->cols(), want.cols());
  for (index_t r = 0; r < want.rows(); ++r) {
    for (index_t c = 0; c < want.cols(); ++c) {
      BOOST_CHECK_LT(std::abs((*got)(r, c) - want(r, c)), 1e-12);
    }
  }
}

BOOST_AUTO_TEST_CASE(RtImplicitOutputIsSorted) {
  const Matrix a = sample(2, 3, 1);
  Matrix out(3, 2);
  BOOST_REQUIRE((*es::rt::plan("ba"))(a, es::into(out)).has_value());
  BOOST_CHECK_LT((out - a.transpose()).cwiseAbs().maxCoeff(), 1e-12);
}
