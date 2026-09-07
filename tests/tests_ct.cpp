// The compile-time entry point: einsum<"ij,jk->ik">(a, b).  The first
// twenty-two cases are v1's suite, ported name for name -- the macro and the
// hana string are gone, nothing else about what they assert has changed.
#include "einsum/einsum.hpp"

#include <cmath>
#include <vector>

#define BOOST_TEST_MODULE EinsumCompileTimeSuite
#include <boost/test/included/unit_test.hpp>

namespace es = einsum;

BOOST_AUTO_TEST_CASE(EinsumTest_2DMatrix1) {
  std::vector A{0, 1, 2, 3, 4, 5};
  std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};
  auto ein = es::einsum<"ij,jk->ik">(mdA, mdB);
  ein.eval();
  auto res_einsum = ein.get_result_span();

  std::vector res{0, 0, 0, 0, 0, 0, 0, 0};
  std::mdspan<int, std::extents<size_t, 2, 4>> mdres{res.data()};
  for (auto i = 0; i < 2; ++i) {
    for (auto k = 0; k < 4; ++k) {
      mdres[i, k] = 0;
      for (auto j = 0; j < 3; ++j) {
        mdres[i, k] += mdA[i, j] * mdB[j, k];
      }
    }
  }
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 4; ++j) {
      BOOST_CHECK_EQUAL((mdres[i, j]), (res_einsum[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_2DMatrix2) {
  std::vector A{1, 1, 1, 2};
  std::vector B{0, 1, 2, 3};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

  auto a = es::einsum<"ij,jk->ik">(mdA, mdB);
  a.eval();
  auto res_einsum = a.get_result_span();

  std::vector res{0, 0, 0, 0};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdres{res.data()};
  for (auto i = 0; i < 2; ++i) {
    for (auto k = 0; k < 2; ++k) {
      mdres[i, k] = 0;
      for (auto j = 0; j < 2; ++j) {
        mdres[i, k] += mdA[i, j] * mdB[j, k];
      }
    }
  }
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 2; ++j) {
      BOOST_CHECK_EQUAL((mdres[i, j]), (res_einsum[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixMul) {
  std::vector mat1{11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44};
  std::vector mat2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};

  auto ein = es::einsum<"ij,jk->ik">(mdmat1, mdmat2);
  ein.eval();
  auto res = ein.get_result_span();

  std::vector res_calc{130, 130, 130, 130, 230, 230, 230, 230,
                       330, 330, 330, 330, 430, 430, 430, 430};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdmatres[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_HadamardProduct) {
  std::vector mat1{11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44};
  std::vector mat2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};

  auto ein = es::einsum<"ij,ij->ij">(mdmat1, mdmat2);
  ein.eval();
  auto res = ein.get_result_span();

  std::vector res_calc{11, 12, 13, 14, 42, 44, 46, 48, 93, 96, 99, 102, 164, 168, 172, 176};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdmatres[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_HadamardProduct2) {
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::vector res_calc{5, 12, 21, 32};

  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdmatres{res_calc.data()};

  auto ein = es::einsum<"ij,ij->ij">(mdA, mdB);
  ein.eval();
  auto result = ein.get_result_span();

  for (auto i = 0; i < 2; i++) {
    for (auto j = 0; j < 2; j++) {
      BOOST_CHECK_EQUAL((result[i, j]), (mdmatres[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixTranspose) {
  std::vector A{1, 2, 3, 4};
  std::vector<int> B{1, 2, 3, 4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{B.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{A.data()};

  // explicit output required: "ij,ji" auto-infers to scalar (numpy semantics)
  auto a = es::einsum<"ij,ji->ij">(mdA, mdB);
  a.eval();
  auto res = a.get_result_span();

  std::vector res_calc{1, 6, 6, 16};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 2; i++) {
    for (auto j = 0; j < 2; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdmatres[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_ElementWiseSquaring) {
  std::vector mat{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat{mat.data()};

  auto ein = es::einsum<"ij,ij->ij">(mdmat, mdmat);
  ein.eval();
  auto res = ein.get_result_span();

  std::vector expected{1, 1, 1, 1, 4, 4, 4, 4, 9, 9, 9, 9, 16, 16, 16, 16};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdexpected{expected.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdexpected[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_DotProduct) {
  std::vector A{1, 2, 3};
  std::vector B{4, 5, 6};
  std::mdspan<int, std::extents<size_t, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3>> mdB{B.data()};
  auto ein = es::einsum<"i,i->">(mdA, mdB);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 32);
}

BOOST_AUTO_TEST_CASE(EinsumTest_OuterProduct) {
  std::vector A{1, 2, 3};
  std::vector B{4, 5};
  std::mdspan<int, std::extents<size_t, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2>> mdB{B.data()};
  auto ein = es::einsum<"i,j->ij">(mdA, mdB);
  ein.eval();
  auto res = ein.get_result_span();
  std::vector expected{4, 5, 8, 10, 12, 15};
  std::mdspan<int, std::extents<size_t, 3, 2>> mdexp{expected.data()};
  for (auto i = 0; i < 3; ++i)
    for (auto j = 0; j < 2; ++j)
      BOOST_CHECK_EQUAL((res[i, j]), (mdexp[i, j]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixVectorMul) {
  std::vector A{1, 2, 3, 4, 5, 6};
  std::vector B{1, 2, 3};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3>> mdB{B.data()};
  auto ein = es::einsum<"ij,j->i">(mdA, mdB);
  ein.eval();
  const auto &res = ein.get_result();
  BOOST_CHECK_EQUAL(res[0], 14);
  BOOST_CHECK_EQUAL(res[1], 32);
}

BOOST_AUTO_TEST_CASE(EinsumTest_FrobeniusInnerProduct) {
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  auto ein = es::einsum<"ij,ij->">(mdA, mdB);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 70);
}

BOOST_AUTO_TEST_CASE(EinsumTest_EvalIdempotent) {
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  auto ein = es::einsum<"ij,jk->ik">(mdA, mdB);
  ein.eval();
  auto first = ein.get_result();
  ein.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(first.begin(), first.end(), ein.get_result().begin(),
                                ein.get_result().end());
}

BOOST_AUTO_TEST_CASE(EinsumTest_AutoInferenceMatchesExplicit) {
  // "ij,jk" auto-infers to "ij,jk->ik" (j repeated -> contracted, i and k unique -> output)
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  auto ein_auto = es::einsum<"ij,jk">(mdA, mdB);
  auto ein_explicit = es::einsum<"ij,jk->ik">(mdA, mdB);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(ein_auto.get_result().begin(), ein_auto.get_result().end(),
                                ein_explicit.get_result().begin(),
                                ein_explicit.get_result().end());
}

// -- Unary einsum tests -------------------------------------------------------

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryTranspose) {
  std::vector A{1, 2, 3, 4, 5, 6};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  auto ein = es::einsum<"ij->ji">(mdA);
  ein.eval();
  auto res = ein.get_result_span();
  std::vector expected{1, 4, 2, 5, 3, 6};
  std::mdspan<int, std::extents<size_t, 3, 2>> mdexp{expected.data()};
  for (auto j = 0; j < 3; ++j)
    for (auto i = 0; i < 2; ++i)
      BOOST_CHECK_EQUAL((res[j, i]), (mdexp[j, i]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryTrace) {
  std::vector A{1, 0, 0, 0, 2, 0, 0, 0, 3};
  std::mdspan<int, std::extents<size_t, 3, 3>> mdA{A.data()};
  auto ein = es::einsum<"ii->">(mdA);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 6);
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryDiagonal) {
  std::vector A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::mdspan<int, std::extents<size_t, 3, 3>> mdA{A.data()};
  auto ein = es::einsum<"ii->i">(mdA);
  ein.eval();
  const auto &res = ein.get_result();
  BOOST_CHECK_EQUAL(res[0], 1);
  BOOST_CHECK_EQUAL(res[1], 5);
  BOOST_CHECK_EQUAL(res[2], 9);
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAxisPermutation) {
  std::vector A{1, 2, 3, 4, 5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2, 2>> mdA{A.data()};
  auto ein = es::einsum<"ijk->kij">(mdA);
  ein.eval();
  auto res = ein.get_result_span();
  for (auto i = 0; i < 2; ++i)
    for (auto j = 0; j < 2; ++j)
      for (auto k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL((res[k, i, j]), (mdA[i, j, k]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAutoInferIdentity) {
  std::vector A{1, 2, 3, 4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  auto ein_auto = es::einsum<"ij">(mdA);
  auto ein_explicit = es::einsum<"ij->ij">(mdA);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(ein_auto.get_result().begin(), ein_auto.get_result().end(),
                                ein_explicit.get_result().begin(),
                                ein_explicit.get_result().end());
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAutoInferTrace) {
  std::vector A{1, 0, 0, 2};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  auto ein_auto = es::einsum<"ii">(mdA);
  auto ein_explicit = es::einsum<"ii->">(mdA);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(ein_auto.get_result().begin(), ein_auto.get_result().end(),
                                ein_explicit.get_result().begin(),
                                ein_explicit.get_result().end());
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryMatMulChain) {
  std::vector A{1, 2, 3, 4, 5, 6};
  std::vector B{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  std::vector C{1, 0, 0, 1, 0, 0, 1, 0};

  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};
  std::mdspan<int, std::extents<size_t, 4, 2>> mdC{C.data()};

  auto ein = es::einsum<"ij,jk,kl->il">(mdA, mdB, mdC);
  ein.eval();
  auto res = ein.get_result_span();

  std::vector expected(4, 0);
  std::mdspan<int, std::extents<size_t, 2, 2>> mdexp{expected.data()};
  for (auto i = 0; i < 2; ++i)
    for (auto l = 0; l < 2; ++l) {
      mdexp[i, l] = 0;
      for (auto j = 0; j < 3; ++j)
        for (auto k = 0; k < 4; ++k)
          mdexp[i, l] += mdA[i, j] * mdB[j, k] * mdC[k, l];
    }
  for (auto i = 0; i < 2; ++i)
    for (auto l = 0; l < 2; ++l)
      BOOST_CHECK_EQUAL((res[i, l]), (mdexp[i, l]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryBatchedDot) {
  std::vector v{1, 0, 0, 1};
  std::vector M{1, 0, 0, 1};

  std::mdspan<int, std::extents<size_t, 2, 2>> mdV{v.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdM{M.data()};

  auto ein = es::einsum<"bi,ij,bj->">(mdV, mdM, mdV);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 2);
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryAutoInfer) {
  std::vector A{1, 2, 3, 4};
  std::vector B{1, 0, 0, 1};
  std::vector C{1, 2, 3, 4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdC{C.data()};
  auto ein_auto = es::einsum<"ij,jk,kl">(mdA, mdB, mdC);
  auto ein_explicit = es::einsum<"ij,jk,kl->il">(mdA, mdB, mdC);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(ein_auto.get_result().begin(), ein_auto.get_result().end(),
                                ein_explicit.get_result().begin(),
                                ein_explicit.get_result().end());
}

// -- v2 additions -------------------------------------------------------------

BOOST_AUTO_TEST_CASE(EinsumCt_DoubleMatmulMatchesEigen) {
  std::vector<double> a(8 * 9);
  std::vector<double> b(9 * 7);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = std::sin(static_cast<double>(i));
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = std::cos(static_cast<double>(i));
  std::mdspan<double, std::extents<size_t, 8, 9>> mdA{a.data()};
  std::mdspan<double, std::extents<size_t, 9, 7>> mdB{b.data()};

  auto ein = es::einsum<"ij,jk->ik">(mdA, mdB);
  ein.eval();

  Eigen::Map<const Eigen::Matrix<double, 8, 9, Eigen::RowMajor>> A{a.data()};
  Eigen::Map<const Eigen::Matrix<double, 9, 7, Eigen::RowMajor>> B{b.data()};
  BOOST_CHECK_LT((ein.result_matrix() - (A * B)).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(EinsumCt_FixedEigenOperandsBothOrders) {
  Eigen::Matrix<double, 2, 3> col_major;
  col_major << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> row_major;
  row_major << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  auto ein = es::einsum<"ij,jk->ik">(col_major, row_major);
  ein.eval();
  const Eigen::Matrix<double, 2, 4> want = col_major * row_major;
  BOOST_CHECK_LT((ein.result_matrix() - want).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(EinsumCt_MappedContractionGroup) {
  // "ijk,jkl->il": j and k are one k-group, contiguous on both sides, so both
  // slabs map without a pack.
  std::vector<int> a(2 * 3 * 4);
  std::vector<int> b(3 * 4 * 5);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<int>(i) + 1;
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<int>(i % 7) + 1;
  std::mdspan<int, std::extents<size_t, 2, 3, 4>> mdA{a.data()};
  std::mdspan<int, std::extents<size_t, 3, 4, 5>> mdB{b.data()};

  auto ein = es::einsum<"ijk,jkl->il">(mdA, mdB);
  ein.eval();
  auto res = ein.get_result_span();
  for (int i = 0; i < 2; ++i)
    for (int l = 0; l < 5; ++l) {
      int want = 0;
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k)
          want += (mdA[i, j, k]) * (mdB[j, k, l]);
      BOOST_CHECK_EQUAL((res[i, l]), want);
    }
}

BOOST_AUTO_TEST_CASE(EinsumCt_PackedContractionGroup) {
  // "ikj,jkl->il": the k-group is j then k, which in the left operand is not a
  // nested pair of strides, so the left slab has to be packed.
  std::vector<int> a(2 * 4 * 3);
  std::vector<int> b(3 * 4 * 5);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<int>(i) + 1;
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<int>(i % 7) + 1;
  std::mdspan<int, std::extents<size_t, 2, 4, 3>> mdA{a.data()};
  std::mdspan<int, std::extents<size_t, 3, 4, 5>> mdB{b.data()};

  auto ein = es::einsum<"ikj,jkl->il">(mdA, mdB);
  static_assert(decltype(ein)::scratch_size > 0,
                "the left slab of ikj,jkl->il does not nest and must be packed");
  ein.eval();
  auto res = ein.get_result_span();
  for (int i = 0; i < 2; ++i)
    for (int l = 0; l < 5; ++l) {
      int want = 0;
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k)
          want += (mdA[i, k, j]) * (mdB[j, k, l]);
      BOOST_CHECK_EQUAL((res[i, l]), want);
    }
}

BOOST_AUTO_TEST_CASE(EinsumCt_BatchedMatmul) {
  std::vector A{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector B{1, 0, 0, 1, 1, 0, 0, 1};
  std::mdspan<int, std::extents<size_t, 2, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2, 2>> mdB{B.data()};
  auto ein = es::einsum<"bij,bjk->bik">(mdA, mdB);
  ein.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(ein.get_result().begin(), ein.get_result().end(), A.begin(),
                                A.end());
}

BOOST_AUTO_TEST_CASE(EinsumCt_BatchedDot) {
  std::vector A{1, 2, 3, 4};
  std::vector B{1, 1, 2, 2};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  auto ein = es::einsum<"bi,bi->b">(mdA, mdB);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 3);
  BOOST_CHECK_EQUAL(ein.get_result()[1], 14);
}

BOOST_AUTO_TEST_CASE(EinsumCt_OuterProductRank4) {
  std::vector A{1, 2};
  std::vector B{3, 4};
  std::mdspan<int, std::extents<size_t, 1, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 1>> mdB{B.data()};
  auto ein = es::einsum<"ij,kl->ijkl">(mdA, mdB);
  static_assert(std::same_as<decltype(ein)::extents_type, std::extents<size_t, 1, 2, 2, 1>>);
  ein.eval();
  const std::vector expected{3, 4, 6, 8};
  BOOST_CHECK_EQUAL_COLLECTIONS(ein.get_result().begin(), ein.get_result().end(),
                                expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(EinsumCt_LeadingDiagonal) {
  std::vector<int> a(2 * 2 * 3);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<int>(i) + 1;
  std::mdspan<int, std::extents<size_t, 2, 2, 3>> mdA{a.data()};
  auto ein = es::einsum<"iij->ij">(mdA);
  ein.eval();
  const std::vector expected{1, 2, 3, 10, 11, 12};
  BOOST_CHECK_EQUAL_COLLECTIONS(ein.get_result().begin(), ein.get_result().end(),
                                expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(EinsumCt_TrailingDiagonalSummed) {
  std::vector<int> a(2 * 3 * 3);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<int>(i) + 1;
  std::mdspan<int, std::extents<size_t, 2, 3, 3>> mdA{a.data()};
  auto ein = es::einsum<"ijj->i">(mdA);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 1 + 5 + 9);
  BOOST_CHECK_EQUAL(ein.get_result()[1], 10 + 14 + 18);
}

BOOST_AUTO_TEST_CASE(EinsumCt_PlainMatmulNeedsNoScratch) {
  std::mdspan<int, std::extents<size_t, 4, 4>> mdA{nullptr};
  auto probe = es::einsum<"ij,jk->ik">(mdA, mdA);
  static_assert(decltype(probe)::scratch_size == 0,
                "a plain packed matmul maps both operands and needs no workspace");
  static_assert(decltype(probe)::output_size == 16);
  static_assert(std::same_as<decltype(probe)::extents_type, std::extents<size_t, 4, 4>>);
  static_assert(std::same_as<decltype(probe)::value_type, int>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(EinsumCt_ImplicitOutputIsSorted) {
  // The one behaviour change from v1: an implicit output is NumPy's -- the
  // labels seen exactly once, in ascending order -- so "ba" infers "->ab".
  std::vector A{1, 2, 3, 4, 5, 6};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  auto ein = es::einsum<"ba">(mdA);
  static_assert(std::same_as<decltype(ein)::extents_type, std::extents<size_t, 3, 2>>);
  ein.eval();
  const std::vector expected{1, 4, 2, 5, 3, 6};
  BOOST_CHECK_EQUAL_COLLECTIONS(ein.get_result().begin(), ein.get_result().end(),
                                expected.begin(), expected.end());
}
