// The two front ends held to one grammar.  Every entry is checked three ways:
// by the constexpr parser in a static_assert, by the Boost.Parser one at run
// time, and against each other -- which is the check that matters, since a
// subscript einsum<"..."> accepts and rt::plan() refuses would be two
// languages wearing one name.
#include "einsum/ct/subscripts.hpp"
#include "einsum/rt/parse.hpp"

#include <string_view>
#include <vector>

#define BOOST_TEST_MODULE EinsumParseSuite
#include <boost/test/included/unit_test.hpp>

namespace es = einsum;
using es::errc;
using es::Labels;
using es::Subscripts;

namespace {

[[nodiscard]] constexpr Labels labels(const std::string_view text) noexcept {
  Labels out;
  for (const char c : text) {
    (void)out.push_back(c);
  }
  return out;
}

[[nodiscard]] constexpr Subscripts
expect(const std::initializer_list<std::string_view> operands,
       const std::string_view output, const bool explicit_output) noexcept {
  Subscripts subs;
  for (const std::string_view op : operands) {
    (void)subs.operands.push_back(labels(op));
  }
  subs.output = labels(output);
  subs.explicit_output = explicit_output;
  return subs;
}

// --- the corpus, checked in a constant expression ----------------------------
using es::impl::parse_subscripts;

static_assert(*parse_subscripts("ij,jk->ik") == expect({"ij", "jk"}, "ik", true));
static_assert(*parse_subscripts("ij,jk") == expect({"ij", "jk"}, "ik", false));
static_assert(*parse_subscripts(" i j , j k -> i k ") == expect({"ij", "jk"}, "ik", true));
static_assert(*parse_subscripts("ba") == expect({"ba"}, "ab", false));
static_assert(*parse_subscripts("ab,cd") == expect({"ab", "cd"}, "abcd", false));
static_assert(*parse_subscripts("ii") == expect({"ii"}, "", false));
static_assert(*parse_subscripts("ii->") == expect({"ii"}, "", true));
static_assert(*parse_subscripts("ii->i") == expect({"ii"}, "i", true));
static_assert(*parse_subscripts("i,i->") == expect({"i", "i"}, "", true));

static_assert(failed_with(parse_subscripts(""), errc::no_operands));
static_assert(failed_with(parse_subscripts("   "), errc::no_operands));
static_assert(failed_with(parse_subscripts(","), errc::empty_operand));
static_assert(failed_with(parse_subscripts("ij,"), errc::empty_operand));
static_assert(failed_with(parse_subscripts("ij,,jk"), errc::empty_operand));
static_assert(failed_with(parse_subscripts("ij-k"), errc::bad_syntax));
static_assert(failed_with(parse_subscripts("ij->ik!"), errc::bad_syntax));
static_assert(failed_with(parse_subscripts("i1"), errc::bad_syntax));
static_assert(failed_with(parse_subscripts("i..."), errc::ellipsis_unsupported));
static_assert(failed_with(parse_subscripts("...ij"), errc::ellipsis_unsupported));
static_assert(failed_with(parse_subscripts("abcdefghi"), errc::rank_too_high));
static_assert(failed_with(parse_subscripts("a,b,c,d,e,f,g,h,i"), errc::too_many_operands));
static_assert(failed_with(parse_subscripts("ij->ik"), errc::unknown_output_label));
static_assert(failed_with(parse_subscripts("ij->ii"), errc::repeated_output_label));

// Every subscript the suite exercises, so the runtime pass below covers exactly
// what the static_asserts above do.
constexpr std::string_view kCorpus[]{
    "ij,jk->ik", "ij,jk", " i j , j k -> i k ", "ba",   "ab,cd",
    "ii",        "ii->",  "ii->i",              "i,i->", "i,j->ij",
    "ijk->kij",  "bij,bjk->bik", "ij,jk,kl->il", "a",   "ij->",
    "ij->ji",    "",      "   ",   ",",   "ij,",  "ij,,jk",
    "ij-k",      "ij->ik!", "i1",  "i...", "...ij", "->ij",
    "abcdefghi", "a,b,c,d,e,f,g,h,i", "ij->ik", "ij->ii"};

} // namespace

BOOST_AUTO_TEST_CASE(ParseTest_FrontEndsAgree) {
  for (const std::string_view source : kCorpus) {
    const auto compile_time = parse_subscripts(source);
    const auto run_time = es::rt::parse(source);
    BOOST_TEST_CONTEXT("subscript = '" << source << "'") {
      BOOST_REQUIRE_EQUAL(compile_time.has_value(), run_time.has_value());
      if (compile_time.has_value()) {
        BOOST_CHECK(*compile_time == *run_time);
      } else {
        BOOST_CHECK_EQUAL(es::message(compile_time.error().code),
                          es::message(run_time.error().code));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(ParseTest_RuntimeMatchesTheStaticCorpus) {
  BOOST_CHECK(*es::rt::parse("ij,jk->ik") == expect({"ij", "jk"}, "ik", true));
  BOOST_CHECK(*es::rt::parse("ba") == expect({"ba"}, "ab", false));
  BOOST_CHECK(*es::rt::parse("ab,cd") == expect({"ab", "cd"}, "abcd", false));
  BOOST_CHECK(*es::rt::parse("ii->") == expect({"ii"}, "", true));
}

BOOST_AUTO_TEST_CASE(ParseTest_PlanIsTheSameEitherWay) {
  const auto from_text = es::rt::plan("ij,jk,kl->il");
  const auto from_source = es::impl::build_plan("ij,jk,kl->il");
  BOOST_REQUIRE(from_text.has_value());
  BOOST_REQUIRE(from_source.has_value());
  BOOST_CHECK(*from_text == *from_source);
}

BOOST_AUTO_TEST_CASE(ParseTest_ErrorsFormat) {
  const auto bad = es::rt::plan("ij,,jk");
  BOOST_REQUIRE(!bad.has_value());
  BOOST_CHECK_EQUAL(std::format("{}", bad.error()),
                    "an operand between the commas has no labels");
}
