// The compile-time entry point: einsum<"ij,jk->ik">().  The first twenty-two
// cases are v1's suite, ported name for name -- the macro, the hana string and
// the separate eval()/get_result() step are gone, nothing else about what they
// assert has changed.
//
// Operands here are static-extent mdspans (the view family, which answers an
// mdarray -- over a std::array when, as here, the extents are in the type)
// except where a case is specifically about Eigen or about a nested
// std::array.
#include "einsum/einsum.hpp"

#include <array>
#include <cmath>
#include <experimental/mdarray>
#include <experimental/mdspan>
#include <vector>

#define BOOST_TEST_MODULE EinsumCompileTimeSuite
#include <boost/test/included/unit_test.hpp>

namespace es = einsum;

namespace {

// Reads an element of a result whatever family it came back in: an mdarray
// takes [i, j], a nest takes [i][j], an Eigen matrix takes (i, j).  The library
// already has one function that knows all three, so the cases below do not have
// to care which family the subscript happened to produce.
template <typename A, typename... I>
[[nodiscard]] auto at(const A &a, const I... idx) {
  return es::impl::element_at(
      a, std::array<es::index_t, sizeof...(I)>{static_cast<es::index_t>(idx)...});
}

// The same, against the values a hand-written loop would produce.
template <typename A, typename T>
[[nodiscard]] bool same(const A &got, const std::vector<std::vector<T>> &want) {
  for (std::size_t i = 0; i < want.size(); ++i) {
    for (std::size_t j = 0; j < want[i].size(); ++j) {
      if (!(at(got, i, j) == want[i][j])) {
        return false;
      }
    }
  }
  return true;
}

template <typename A, typename T>
[[nodiscard]] bool same(const A &got,
                        const std::vector<std::vector<std::vector<T>>> &want) {
  for (std::size_t i = 0; i < want.size(); ++i) {
    for (std::size_t j = 0; j < want[i].size(); ++j) {
      for (std::size_t k = 0; k < want[i][j].size(); ++k) {
        if (!(at(got, i, j, k) == want[i][j][k])) {
          return false;
        }
      }
    }
  }
  return true;
}

// Two results of one call shape, compared over the elements they hold.
template <typename A> [[nodiscard]] bool same_as(const A &a, const A &b) {
  return std::equal(a.data(), a.data() + a.size(), b.data());
}

template <std::size_t... E, typename T>
[[nodiscard]] auto view_of(const std::vector<T> &v) {
  return std::mdspan<const T, std::extents<std::size_t, E...>>{v.data()};
}

} // namespace

// --- "..."_ct: the subscript as an argument -----------------------------------
// A function argument is never a constant expression, so the suffix is what
// puts the subscript in the template parameter it has to reach.  What comes
// back is the same object the angle-bracket spelling answers -- asserted here,
// because two spellings of one thing that differ are worse than one spelling.
BOOST_AUTO_TEST_CASE(EinsumCt_LiteralSuffixIsTheSameObject) {
  using einsum::literals::operator""_ct;

  const auto suffixed = es::einsum("ij,jk->ik"_ct);
  const auto angled = es::einsum<"ij,jk->ik">();
  static_assert(std::same_as<decltype(suffixed), decltype(angled)>);

  const auto ordered = es::einsum("ab,bc,cd->ad"_ct.with<es::path::sequential>());
  static_assert(std::same_as<decltype(ordered),
                             const decltype(es::einsum<"ab,bc,cd->ad",
                                                       es::path::sequential>())>);
  BOOST_CHECK_EQUAL(ordered.operand_count(), 3U);

  const std::vector A{0, 1, 2, 3, 4, 5};
  const std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const auto res = suffixed(view_of<2, 3>(A), view_of<3, 4>(B));
  BOOST_REQUIRE(res.has_value());
  const auto same = angled(view_of<2, 3>(A), view_of<3, 4>(B));
  BOOST_REQUIRE(same.has_value());
  BOOST_CHECK_EQUAL(at(*res, 1, 2), at(*same, 1, 2));
}

BOOST_AUTO_TEST_CASE(EinsumTest_2DMatrix1) {
  const std::vector A{0, 1, 2, 3, 4, 5};
  const std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const auto mdA = view_of<2, 3>(A);
  const auto mdB = view_of<3, 4>(B);
  const auto res = es::einsum<"ij,jk->ik">()(mdA, mdB);
  BOOST_REQUIRE(res.has_value());
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t k = 0; k < 4; ++k) {
      int want = 0;
      for (std::size_t j = 0; j < 3; ++j) {
        want += (mdA[i, j]) * (mdB[j, k]);
      }
      BOOST_CHECK_EQUAL(at(*res, i, k), want);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_2DMatrix2) {
  const std::vector A{1, 1, 1, 2};
  const std::vector B{0, 1, 2, 3};
  const auto res =
      es::einsum<"ij,jk->ik">()(view_of<2, 2>(A), view_of<2, 2>(B));
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<int>> want{{2, 4}, {4, 7}};
  BOOST_CHECK(same(*res, want));
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixMul) {
  const std::vector m1{11, 12, 13, 14, 21, 22, 23, 24,
                       31, 32, 33, 34, 41, 42, 43, 44};
  const std::vector m2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  const auto res =
      es::einsum<"ij,jk->ik">()(view_of<4, 4>(m1), view_of<4, 4>(m2));
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<int>> want{{130, 130, 130, 130},
                                           {230, 230, 230, 230},
                                           {330, 330, 330, 330},
                                           {430, 430, 430, 430}};
  BOOST_CHECK(same(*res, want));
}

BOOST_AUTO_TEST_CASE(EinsumTest_HadamardProduct) {
  const std::vector m1{11, 12, 13, 14, 21, 22, 23, 24,
                       31, 32, 33, 34, 41, 42, 43, 44};
  const std::vector m2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  const auto res =
      es::einsum<"ij,ij->ij">()(view_of<4, 4>(m1), view_of<4, 4>(m2));
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<int>> want{{11, 12, 13, 14},
                                           {42, 44, 46, 48},
                                           {93, 96, 99, 102},
                                           {164, 168, 172, 176}};
  BOOST_CHECK(same(*res, want));
}

BOOST_AUTO_TEST_CASE(EinsumTest_HadamardProduct2) {
  const std::vector A{1, 2, 3, 4};
  const std::vector B{5, 6, 7, 8};
  const auto res =
      es::einsum<"ij,ij->ij">()(view_of<2, 2>(A), view_of<2, 2>(B));
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<int>> want{{5, 12}, {21, 32}};
  BOOST_CHECK(same(*res, want));
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixTranspose) {
  const std::vector A{1, 2, 3, 4};
  // explicit output required: "ij,ji" auto-infers to a scalar (numpy semantics)
  const auto res =
      es::einsum<"ij,ji->ij">()(view_of<2, 2>(A), view_of<2, 2>(A));
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<int>> want{{1, 6}, {6, 16}};
  BOOST_CHECK(same(*res, want));
}

BOOST_AUTO_TEST_CASE(EinsumTest_ElementWiseSquaring) {
  const std::vector m{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  const auto md = view_of<4, 4>(m);
  const auto res = es::einsum<"ij,ij->ij">()(md, md);
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<int>> want{
      {1, 1, 1, 1}, {4, 4, 4, 4}, {9, 9, 9, 9}, {16, 16, 16, 16}};
  BOOST_CHECK(same(*res, want));
}

BOOST_AUTO_TEST_CASE(EinsumTest_DotProduct) {
  const std::vector A{1, 2, 3};
  const std::vector B{4, 5, 6};
  const auto res = es::einsum<"i,i->">()(view_of<3>(A), view_of<3>(B));
  BOOST_REQUIRE(res.has_value());
  // A rank-0 result in a rank-1 family is one element behind one leading axis.
  BOOST_CHECK_EQUAL(at(*res), 32);
}

BOOST_AUTO_TEST_CASE(EinsumTest_OuterProduct) {
  const std::vector A{1, 2, 3};
  const std::vector B{4, 5};
  // rank 2 out of rank-1 operands: the by-value form cannot reach it, so the
  // output names the rank instead.
  std::vector<std::vector<int>> res;
  BOOST_REQUIRE(
      es::einsum<"i,j->ij">()(view_of<3>(A), view_of<2>(B), res).has_value());
  const std::vector<std::vector<int>> want{{4, 5}, {8, 10}, {12, 15}};
  BOOST_CHECK(res == want);
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixVectorMul) {
  const std::vector A{1, 2, 3, 4, 5, 6};
  const std::vector B{1, 2, 3};
  const auto res = es::einsum<"ij,j->i">()(view_of<2, 3>(A), view_of<3>(B));
  BOOST_REQUIRE(res.has_value());
  BOOST_CHECK_EQUAL(at(*res, 0), 14);
  BOOST_CHECK_EQUAL(at(*res, 1), 32);
}

BOOST_AUTO_TEST_CASE(EinsumTest_FrobeniusInnerProduct) {
  const std::vector A{1, 2, 3, 4};
  const std::vector B{5, 6, 7, 8};
  const auto res = es::einsum<"ij,ij->">()(view_of<2, 2>(A), view_of<2, 2>(B));
  BOOST_REQUIRE(res.has_value());
  BOOST_CHECK_EQUAL(at(*res), 70);
}

BOOST_AUTO_TEST_CASE(EinsumTest_EvalIdempotent) {
  const std::vector A{1, 2, 3, 4};
  const std::vector B{5, 6, 7, 8};
  const auto e = es::einsum<"ij,jk->ik">();
  const auto first = e(view_of<2, 2>(A), view_of<2, 2>(B));
  const auto second = e(view_of<2, 2>(A), view_of<2, 2>(B));
  BOOST_REQUIRE(first.has_value() && second.has_value());
  BOOST_CHECK(same_as(*first, *second));
}

BOOST_AUTO_TEST_CASE(EinsumTest_AutoInferenceMatchesExplicit) {
  // "ij,jk" auto-infers to "ij,jk->ik"
  const std::vector A{1, 2, 3, 4};
  const std::vector B{5, 6, 7, 8};
  const auto inferred =
      es::einsum<"ij,jk">()(view_of<2, 2>(A), view_of<2, 2>(B));
  const auto explicitly =
      es::einsum<"ij,jk->ik">()(view_of<2, 2>(A), view_of<2, 2>(B));
  BOOST_REQUIRE(inferred.has_value() && explicitly.has_value());
  BOOST_CHECK(same_as(*inferred, *explicitly));
}

// -- Unary einsum tests -------------------------------------------------------

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryTranspose) {
  const std::vector A{1, 2, 3, 4, 5, 6};
  const auto res = es::einsum<"ij->ji">()(view_of<2, 3>(A));
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<int>> want{{1, 4}, {2, 5}, {3, 6}};
  BOOST_CHECK(same(*res, want));
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryTrace) {
  const std::vector A{1, 0, 0, 0, 2, 0, 0, 0, 3};
  const auto res = es::einsum<"ii->">()(view_of<3, 3>(A));
  BOOST_REQUIRE(res.has_value());
  BOOST_CHECK_EQUAL(at(*res), 6);
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryDiagonal) {
  const std::vector A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const auto res = es::einsum<"ii->i">()(view_of<3, 3>(A));
  BOOST_REQUIRE(res.has_value());
  BOOST_CHECK_EQUAL(at(*res, 0), 1);
  BOOST_CHECK_EQUAL(at(*res, 1), 5);
  BOOST_CHECK_EQUAL(at(*res, 2), 9);
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAxisPermutation) {
  const std::vector A{1, 2, 3, 4, 5, 6, 7, 8};
  const auto mdA = view_of<2, 2, 2>(A);
  const auto res = es::einsum<"ijk->kij">()(mdA);
  BOOST_REQUIRE(res.has_value());
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      for (std::size_t k = 0; k < 2; ++k) {
        BOOST_CHECK_EQUAL(at(*res, k, i, j), (mdA[i, j, k]));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAutoInferIdentity) {
  const std::vector A{1, 2, 3, 4};
  const auto inferred = es::einsum<"ij">()(view_of<2, 2>(A));
  const auto explicitly = es::einsum<"ij->ij">()(view_of<2, 2>(A));
  BOOST_REQUIRE(inferred.has_value() && explicitly.has_value());
  BOOST_CHECK(same_as(*inferred, *explicitly));
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAutoInferTrace) {
  const std::vector A{1, 0, 0, 2};
  const auto inferred = es::einsum<"ii">()(view_of<2, 2>(A));
  const auto explicitly = es::einsum<"ii->">()(view_of<2, 2>(A));
  BOOST_REQUIRE(inferred.has_value() && explicitly.has_value());
  BOOST_CHECK(same_as(*inferred, *explicitly));
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryMatMulChain) {
  const std::vector A{1, 2, 3, 4, 5, 6};
  const std::vector B{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  const std::vector C{1, 0, 0, 1, 0, 0, 1, 0};
  const auto mdA = view_of<2, 3>(A);
  const auto mdB = view_of<3, 4>(B);
  const auto mdC = view_of<4, 2>(C);
  const auto res = es::einsum<"ij,jk,kl->il">()(mdA, mdB, mdC);
  BOOST_REQUIRE(res.has_value());
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t l = 0; l < 2; ++l) {
      int want = 0;
      for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t k = 0; k < 4; ++k) {
          want += (mdA[i, j]) * (mdB[j, k]) * (mdC[k, l]);
        }
      }
      BOOST_CHECK_EQUAL(at(*res, i, l), want);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryBatchedDot) {
  const std::vector v{1, 0, 0, 1};
  const std::vector M{1, 0, 0, 1};
  const auto res = es::einsum<"bi,ij,bj->">()(
      view_of<2, 2>(v), view_of<2, 2>(M), view_of<2, 2>(v));
  BOOST_REQUIRE(res.has_value());
  BOOST_CHECK_EQUAL(at(*res), 2);
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryAutoInfer) {
  const std::vector A{1, 2, 3, 4};
  const std::vector B{1, 0, 0, 1};
  const std::vector C{1, 2, 3, 4};
  const auto inferred = es::einsum<"ij,jk,kl">()(
      view_of<2, 2>(A), view_of<2, 2>(B), view_of<2, 2>(C));
  const auto explicitly = es::einsum<"ij,jk,kl->il">()(
      view_of<2, 2>(A), view_of<2, 2>(B), view_of<2, 2>(C));
  BOOST_REQUIRE(inferred.has_value() && explicitly.has_value());
  BOOST_CHECK(same_as(*inferred, *explicitly));
}

// -- v2 additions -------------------------------------------------------------

BOOST_AUTO_TEST_CASE(EinsumCt_DoubleMatmulMatchesEigen) {
  std::vector<double> a(8 * 9);
  std::vector<double> b(9 * 7);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = std::sin(static_cast<double>(i));
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = std::cos(static_cast<double>(i));
  }
  const auto res =
      es::einsum<"ij,jk->ik">()(view_of<8, 9>(a), view_of<9, 7>(b));
  BOOST_REQUIRE(res.has_value());

  const Eigen::Map<const Eigen::Matrix<double, 8, 9, Eigen::RowMajor>> A{
      a.data()};
  const Eigen::Map<const Eigen::Matrix<double, 9, 7, Eigen::RowMajor>> B{
      b.data()};
  const Eigen::Matrix<double, 8, 7> want = A * B;
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t k = 0; k < 7; ++k) {
      BOOST_CHECK_LT(std::abs(at(*res, i, k) -
                              want(static_cast<int>(i), static_cast<int>(k))),
                     1e-12);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumCt_FixedEigenOperandsBothOrders) {
  Eigen::Matrix<double, 2, 3> col_major;
  col_major << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> row_major;
  row_major << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  // Different Eigen types, one family: the column-major operand goes through
  // the same path as the row-major one.
  const auto res = es::einsum<"ij,jk->ik">()(col_major, row_major);
  BOOST_REQUIRE(res.has_value());
  const Eigen::Matrix<double, 2, 4> want = col_major * row_major;
  BOOST_CHECK_LT((*res - want).cwiseAbs().maxCoeff(), 1e-12);
}

BOOST_AUTO_TEST_CASE(EinsumCt_MappedContractionGroup) {
  // "ijk,jkl->il": j and k are one k-group, contiguous on both sides.
  std::vector<int> a(2 * 3 * 4);
  std::vector<int> b(3 * 4 * 5);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<int>(i) + 1;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<int>(i % 7) + 1;
  }
  const auto mdA = view_of<2, 3, 4>(a);
  const auto mdB = view_of<3, 4, 5>(b);
  const auto res = es::einsum<"ijk,jkl->il">()(mdA, mdB);
  BOOST_REQUIRE(res.has_value());
  // rank-3 operands, rank-2 output: one leading axis of 1.
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t l = 0; l < 5; ++l) {
      int want = 0;
      for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t k = 0; k < 4; ++k) {
          want += (mdA[i, j, k]) * (mdB[j, k, l]);
        }
      }
      BOOST_CHECK_EQUAL(at(*res, i, l), want);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumCt_PackedContractionGroup) {
  // "ikj,jkl->il": the k-group is j then k, which in the left operand is not a
  // nested pair of strides, so the left slab has to be packed.
  std::vector<int> a(2 * 4 * 3);
  std::vector<int> b(3 * 4 * 5);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<int>(i) + 1;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<int>(i % 7) + 1;
  }
  const auto mdA = view_of<2, 4, 3>(a);
  const auto mdB = view_of<3, 4, 5>(b);
  const auto res = es::einsum<"ikj,jkl->il">()(mdA, mdB);
  BOOST_REQUIRE(res.has_value());
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t l = 0; l < 5; ++l) {
      int want = 0;
      for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t k = 0; k < 4; ++k) {
          want += (mdA[i, k, j]) * (mdB[j, k, l]);
        }
      }
      BOOST_CHECK_EQUAL(at(*res, i, l), want);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumCt_BatchedMatmul) {
  const std::vector A{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector B{1, 0, 0, 1, 1, 0, 0, 1};
  const auto res =
      es::einsum<"bij,bjk->bik">()(view_of<2, 2, 2>(A), view_of<2, 2, 2>(B));
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<std::vector<int>>> want{{{1, 2}, {3, 4}},
                                                        {{5, 6}, {7, 8}}};
  BOOST_CHECK(same(*res, want));
}

BOOST_AUTO_TEST_CASE(EinsumCt_BatchedDot) {
  const std::vector A{1, 2, 3, 4};
  const std::vector B{1, 1, 2, 2};
  const auto res = es::einsum<"bi,bi->b">()(view_of<2, 2>(A), view_of<2, 2>(B));
  BOOST_REQUIRE(res.has_value());
  BOOST_CHECK_EQUAL(at(*res, 0), 3);
  BOOST_CHECK_EQUAL(at(*res, 1), 14);
}

BOOST_AUTO_TEST_CASE(EinsumCt_OuterProductRank4) {
  const std::vector A{1, 2};
  const std::vector B{3, 4};
  // Rank 4 out of rank-2 operands.  The compile-time path knows the output
  // shape exactly, so it can hand back a rank-4 result by value -- which the
  // runtime path cannot, because there the return type cannot see the
  // subscript.
  const auto res = es::einsum<"ij,kl->ijkl">()(view_of<1, 2>(A), view_of<2, 1>(B));
  BOOST_REQUIRE(res.has_value());
  static_assert(std::remove_cvref_t<decltype(*res)>::rank() == 4);
  BOOST_CHECK_EQUAL(at(*res, 0, 0, 0, 0), 3);
  BOOST_CHECK_EQUAL(at(*res, 0, 0, 1, 0), 4);
  BOOST_CHECK_EQUAL(at(*res, 0, 1, 0, 0), 6);
  BOOST_CHECK_EQUAL(at(*res, 0, 1, 1, 0), 8);

  // And the output form still names the rank itself, in whatever family the
  // caller wants it.
  std::vector<std::vector<std::vector<std::vector<int>>>> nested;
  BOOST_REQUIRE(
      es::einsum<"ij,kl->ijkl">()(view_of<1, 2>(A), view_of<2, 1>(B), nested).has_value());
  BOOST_CHECK_EQUAL(nested[0][1][1][0], 8);
}

BOOST_AUTO_TEST_CASE(EinsumCt_LeadingDiagonal) {
  std::vector<int> a(2 * 2 * 3);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<int>(i) + 1;
  }
  const auto res = es::einsum<"iij->ij">()(view_of<2, 2, 3>(a));
  BOOST_REQUIRE(res.has_value());
  BOOST_CHECK_EQUAL(at(*res, 0, 0), 1);
  BOOST_CHECK_EQUAL(at(*res, 0, 2), 3);
  BOOST_CHECK_EQUAL(at(*res, 1, 0), 10);
  BOOST_CHECK_EQUAL(at(*res, 1, 2), 12);
}

BOOST_AUTO_TEST_CASE(EinsumCt_TrailingDiagonalSummed) {
  std::vector<int> a(2 * 3 * 3);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<int>(i) + 1;
  }
  const auto res = es::einsum<"ijj->i">()(view_of<2, 3, 3>(a));
  BOOST_REQUIRE(res.has_value());
  BOOST_CHECK_EQUAL(at(*res, 0), 1 + 5 + 9);
  BOOST_CHECK_EQUAL(at(*res, 1), 10 + 14 + 18);
}

BOOST_AUTO_TEST_CASE(EinsumCt_PlainMatmulNeedsNoScratch) {
  // A nested std::array is an owning family, so the result is the same type.
  const std::array<std::array<int, 2>, 2> a{{{1, 2}, {3, 4}}};
  const auto res = es::einsum<"ij,jk->ik">()(a, a);
  BOOST_REQUIRE(res.has_value());
  static_assert(std::same_as<std::remove_cvref_t<decltype(*res)>,
                             std::array<std::array<int, 2>, 2>>);
  BOOST_CHECK_EQUAL(at(*res, 0, 0), 7);
  BOOST_CHECK_EQUAL(at(*res, 1, 1), 22);
  // operand_count() is static, so it is a constant without an object -- the
  // object itself is not a literal type, because its scratch cache is mutable.
  static_assert(es::StaticEinsum<"ij,jk->ik">::operand_count() == 2);
}

BOOST_AUTO_TEST_CASE(EinsumCt_ImplicitOutputIsSorted) {
  // An implicit output is NumPy's -- the labels seen exactly once, in ascending
  // order -- so "ba" infers "->ab".
  const std::vector A{1, 2, 3, 4, 5, 6};
  const auto res = es::einsum<"ba">()(view_of<2, 3>(A));
  BOOST_REQUIRE(res.has_value());
  const std::vector<std::vector<int>> want{{1, 4}, {2, 5}, {3, 6}};
  BOOST_CHECK(same(*res, want));
}

// The output form, which only this path can have: the operand count is a
// constant here, so a third argument is unambiguously an output even when the
// operands are non-const.
BOOST_AUTO_TEST_CASE(EinsumCt_OutputFormIsUnambiguous) {
  Eigen::MatrixXd a(2, 3);
  Eigen::MatrixXd b(3, 2);
  a << 1, 2, 3, 4, 5, 6;
  b << 1, 0, 0, 1, 0, 0;
  const auto e = es::einsum<"ij,jk->ik">();

  const auto by_value = e(a, b);
  static_assert(std::same_as<std::remove_cvref_t<decltype(by_value)>,
                             es::result<Eigen::MatrixXd>>);
  BOOST_REQUIRE(by_value.has_value());

  Eigen::MatrixXd out;
  const auto written = e(a, b, out);
  static_assert(
      std::same_as<std::remove_cvref_t<decltype(written)>, es::result<void>>);
  BOOST_REQUIRE(written.has_value());
  BOOST_CHECK_LT((out - a * b).cwiseAbs().maxCoeff(), 1e-12);
  BOOST_CHECK_LT((out - *by_value).cwiseAbs().maxCoeff(), 1e-12);
}

// --- the constant lowering ---------------------------------------------------
// When every extent is in a type the whole lowering is a constant: the shape,
// the output layout, the Geometry and the scratch size.  Whether a given call
// then reaches Eigen's unrolled fixed-size product is a separate question, and
// both are answered here without running anything.
namespace {

using MdRight = std::mdspan<const double, std::extents<std::size_t, 3, 3>>;
using FixedRow = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using FixedCol = Eigen::Matrix<double, 3, 3>; // Eigen's default is column-major
using ArrayNest = std::array<std::array<double, 3>, 3>;
using PackA = std::mdspan<const double, std::extents<std::size_t, 2, 4, 3>>;
using PackB = std::mdspan<const double, std::extents<std::size_t, 3, 4, 5>>;

using MatMul = es::StaticEinsum<"ij,jk->ik">;
using Packed = es::StaticEinsum<"ikj,jkl->il">;

} // namespace

BOOST_AUTO_TEST_CASE(EinsumCt_StaticOperandsLowerAtCompileTime) {
  static_assert(MatMul::lowers_statically<MdRight, MdRight>());
  static_assert(MatMul::lowers_statically<FixedRow, FixedRow>());
  static_assert(MatMul::lowers_statically<ArrayNest, ArrayNest>());
  static_assert(Packed::lowers_statically<PackA, PackB>());
  // A dynamic extent anywhere and there is nothing to know early.
  static_assert(!MatMul::lowers_statically<Eigen::MatrixXd, Eigen::MatrixXd>());
  static_assert(!MatMul::lowers_statically<
                std::mdspan<const double, std::dextents<std::size_t, 2>>,
                std::mdspan<const double, std::dextents<std::size_t, 2>>>());

  // The view family's compile-time result: an mdarray whose extents are in its
  // type, over a std::array, so the call allocates nothing at all.
  const std::vector<double> raw(9, 1.0);
  const auto got = es::einsum<"ij,jk->ik">()(MdRight{raw.data()}, MdRight{raw.data()});
  BOOST_REQUIRE(got.has_value());
  static_assert(
      std::same_as<std::remove_cvref_t<decltype(*got)>,
                   std::experimental::mdarray<double, std::extents<std::size_t, 3, 3>,
                                              std::layout_right, std::array<double, 9>>>);
  BOOST_CHECK_LT(std::abs(at(*got, 0, 0) - 3.0), 1e-12);
}

BOOST_AUTO_TEST_CASE(EinsumCt_UnrolledProductIsTakenExactlyWhereItApplies) {
  // Row-major fixed Eigen: one mappable GEMM writing into a row-major result.
  static_assert(MatMul::unrolls<FixedRow, FixedRow>());

  // A contraction that has to pack a slab is not one plain GEMM.
  static_assert(!Packed::unrolls<PackA, PackB>());
  // Neither is a reduction.
  static_assert(!es::StaticEinsum<"ii->">::unrolls<MdRight>());
  // Nor anything whose extents are not all known.
  static_assert(!MatMul::unrolls<Eigen::MatrixXd, Eigen::MatrixXd>());

  // Column-major fixed Eigen does too, which matters because that is Eigen's
  // default order: a column-major A times a column-major B into a column-major
  // C is a plain fixed product, and so is any mixture of the two orders.
  static_assert(MatMul::unrolls<FixedCol, FixedCol>());
  static_assert(MatMul::unrolls<FixedCol, FixedRow>());
  static_assert(MatMul::unrolls<FixedRow, FixedCol>());

  // And static-extent mdspan, now that a view's result is a contiguous mdarray
  // the product can be written straight into rather than a nest of vectors.
  static_assert(MatMul::unrolls<MdRight, MdRight>());

  // A std::array nest is not on the contiguous path, so its operands are
  // gathered and the unrolled product does not apply.
  static_assert(!MatMul::unrolls<ArrayNest, ArrayNest>());
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(EinsumCt_UnrolledProductComputesTheSameAnswer) {
  FixedRow a;
  FixedRow b;
  a << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  b << 9, 8, 7, 6, 5, 4, 3, 2, 1;
  const FixedCol ca = a;
  const FixedCol cb = b;
  const Eigen::Matrix<double, 3, 3> want = a * b;

  // Each order, and the two mixtures: the map follows the operand rather than
  // the operand being made to follow the map, so all four agree.
  const auto rr = es::einsum<"ij,jk->ik">()(a, b);
  const auto cc = es::einsum<"ij,jk->ik">()(ca, cb);
  const auto cr = es::einsum<"ij,jk->ik">()(ca, b);
  const auto rc = es::einsum<"ij,jk->ik">()(a, cb);
  BOOST_REQUIRE(rr.has_value() && cc.has_value() && cr.has_value() &&
                rc.has_value());
  BOOST_CHECK_LT((*rr - want).cwiseAbs().maxCoeff(), 1e-12);
  BOOST_CHECK_LT((*cc - want).cwiseAbs().maxCoeff(), 1e-12);
  BOOST_CHECK_LT((*cr - want).cwiseAbs().maxCoeff(), 1e-12);
  BOOST_CHECK_LT((*rc - want).cwiseAbs().maxCoeff(), 1e-12);
}

// --- ellipsis and the path, decided at compile time ---------------------------
namespace {

using MdBatch = std::mdspan<const double, std::extents<std::size_t, 2, 2, 3>>;
using MdRight3 = std::mdspan<const double, std::extents<std::size_t, 2, 3, 2>>;

// The three-operand chain, whose cheap pair depends on which extents are thin.
using Thin0 = std::mdspan<const double, std::extents<std::size_t, 2, 100>>;
using Thin1 = std::mdspan<const double, std::extents<std::size_t, 100, 2>>;
using Thin2 = std::mdspan<const double, std::extents<std::size_t, 2, 100>>;
using Fat0 = std::mdspan<const double, std::extents<std::size_t, 100, 2>>;
using Fat1 = std::mdspan<const double, std::extents<std::size_t, 2, 100>>;
using Fat2 = std::mdspan<const double, std::extents<std::size_t, 100, 2>>;

} // namespace

BOOST_AUTO_TEST_CASE(EinsumCt_EllipsisExpandsAtCompileTime) {
  const std::vector<double> xs(2 * 2 * 3, 2.0);
  const std::vector<double> ys(2 * 3 * 2, 3.0);
  const MdBatch x{xs.data()};
  const MdRight3 y{ys.data()};

  const auto got = es::einsum<"...ij,...jk->...ik">()(x, y);
  BOOST_REQUIRE(got.has_value());
  // The expanded subscript is a constant here, so the result's extents are
  // exact rather than fitted.
  static_assert(std::remove_cvref_t<decltype(*got)>::rank() == 3);
  BOOST_CHECK_EQUAL(got->extent(0), 2U);
  BOOST_CHECK_EQUAL(got->extent(2), 2U);
  for (std::size_t b = 0; b < 2; ++b) {
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t k = 0; k < 2; ++k) {
        BOOST_CHECK_LT(std::abs(at(*got, b, i, k) - 3 * 2.0 * 3.0), 1e-12);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumCt_PathIsChosenAtCompileTime) {
  using Chain = es::StaticEinsum<"ij,jk,kl->il">;
  using Fat = es::StaticEinsum<"ab,bc,cd->ad">;
  using Plain = es::StaticEinsum<"ab,bc,cd->ad", es::path::sequential>;

  // A thin middle: the first pair is the cheap one.
  constexpr es::Path thin =
      es::impl::einsum_access::static_path<Chain, Thin0, Thin1, Thin2>();
  static_assert(thin.steps.size() == 2);
  static_assert(thin.steps[0].l_src == 0 && thin.steps[0].r_src == 1);

  // A fat middle: the last pair is.
  constexpr es::Path fat = es::impl::einsum_access::static_path<Fat, Fat0, Fat1, Fat2>();
  static_assert(fat.steps.size() == 2);
  static_assert(fat.steps[0].l_src == 1 && fat.steps[0].r_src == 2);

  // And the policy is honoured: sequential keeps the subscript's own order even
  // where it is the dearer one.
  constexpr es::Path plain = es::impl::einsum_access::static_path<Plain, Fat0, Fat1, Fat2>();
  static_assert(plain.steps[0].l_src == 0 && plain.steps[0].r_src == 1);
  static_assert(plain.steps[1].l_src == es::kIntermediate && plain.steps[1].r_src == 2);
  BOOST_CHECK(true);
}
