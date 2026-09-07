#include "einsum/rt/parse.hpp"

#include "einsum/core/limits.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"

#include <boost/parser/parser.hpp>

#include <string_view>

// The subscript grammar, and the only translation unit that sees Boost.Parser
// -- which wraps every parse() in a try/catch and so cannot compile under the
// -fno-exceptions the rest of einsum is built with.  The CMakeLists beside this
// file takes the flag off this one object, and the catch below is what keeps
// that local: nothing thrown here reaches a caller.
//
// Everything this file decides that is not the shape of the text -- the
// ellipsis refusal, the implicit output, the explicit output's validity -- it
// decides by calling the same helpers impl::parse_subscripts does, so the two
// front ends cannot drift apart.  tests/tests_parse.cpp checks that they have not.
namespace einsum::rt {
namespace {

namespace bp = boost::parser;

// std::tuple, never boost::hana::tuple.  Nothing here asks for Hana and nothing
// in this project may pull it in; a default that flipped would do so silently.
static_assert(BOOST_PARSER_USE_STD_TUPLE,
              "einsum::rt: Boost.Parser must use std::tuple, not Hana");

using bp::_attr;
using bp::_globals;
using bp::_pass;

// Boost.Parser's own handler writes a caret diagnostic to a stream.  einsum
// reports on the numeric path and nowhere else, so this one says only that the
// parse failed and leaves the errc to say what about it did.
struct Silent {
  template <typename Iter, typename Sentinel>
  constexpr bp::error_handler_result operator()(Iter, Sentinel,
                                                const bp::parse_error<Iter> &) const {
    return bp::error_handler_result::fail;
  }
  template <typename Context>
  constexpr void diagnose(bp::diagnostic_kind, std::string_view, const Context &) const {}
  template <typename Context, typename Iter>
  constexpr void diagnose(bp::diagnostic_kind, std::string_view, const Context &,
                          Iter) const {}
};

// What the actions build.  Carried through with_globals rather than captured:
// BOOST_PARSER_DEFINE_RULES wants its rules at namespace scope, where a lambda
// has nothing to capture.  A synthesized attribute would not do either -- the
// probe that drove this design is that `(operand % ',') >> -("->" >> operand)`
// collapses the optional output straight into the operand vector.
struct Building {
  Subscripts subs{};
  Labels current{};
  bool in_output = false;
  errc why = errc::bad_syntax; // what the refusing action meant, if one did
  bool refused = false;
};

// --- the actions -------------------------------------------------------------
namespace act {

// One refusal shape for every limit: name the code, fail the parser, and let
// the caller read `why` back once the whole parse has failed.
//
// The first refusal wins.  A parser that fails backtracks into whatever the
// grammar offers next, and the alternative's own complaint is about the text it
// was handed rather than about the one that actually went wrong: a ninth
// operand refused as too_many_operands must not be re-reported as an empty one
// by the eps that follows it.
template <errc C> constexpr auto refuse = [](auto &ctx) {
  Building &building = _globals(ctx);
  if (!building.refused) {
    building.why = C;
    building.refused = true;
  }
  _pass(ctx) = false;
};

constexpr auto push_label = [](auto &ctx) {
  Building &building = _globals(ctx);
  Labels &target = building.in_output ? building.subs.output : building.current;
  if (!target.push_back(_attr(ctx))) {
    refuse<errc::rank_too_high>(ctx);
  }
};

constexpr auto end_operand = [](auto &ctx) {
  Building &building = _globals(ctx);
  if (!building.subs.operands.push_back(building.current)) {
    refuse<errc::too_many_operands>(ctx);
    return;
  }
  building.current.clear();
};

// Fires on the arrow itself, so the labels after it land in the output.
constexpr auto begin_output = [](auto &ctx) {
  Building &building = _globals(ctx);
  building.in_output = true;
  building.subs.explicit_output = true;
};

} // namespace act

// --- the grammar -------------------------------------------------------------
// Rules carry no attribute: the actions above write into the globals, and there
// is nothing left for Boost.Parser to synthesize or for a merge rule to reshape.
bp::rule<struct label_tag> const label = "label";
bp::rule<struct operand_tag> const operand = "operand";
bp::rule<struct ellipsis_tag> const ellipsis = "ellipsis";
bp::rule<struct inputs_tag> const inputs = "operands";
bp::rule<struct output_tag> const output = "output";
bp::rule<struct subscript_tag> const subscript = "subscript";

constexpr auto letter = bp::char_('a', 'z') | bp::char_('A', 'Z');

auto const label_def = letter[act::push_label];
auto const operand_def = +label;
auto const ellipsis_def = bp::lit("...")[act::refuse<errc::ellipsis_unsupported>];
// eps after operand: a comma with nothing between it and the next one is an
// empty operand, which is a different complaint from "this is not a subscript".
auto const inputs_def = (operand[act::end_operand] | bp::eps[act::refuse<errc::empty_operand>]) % ',';
auto const output_def = bp::lit("->")[act::begin_output] >> *label;
auto const subscript_def = inputs >> -(ellipsis | output) >> bp::eoi;

BOOST_PARSER_DEFINE_RULES(label, operand, ellipsis, inputs, output, subscript);

} // namespace

result<Subscripts> parse(const std::string_view source) noexcept {
  if (const auto ok = impl::precheck(source); !ok) {
    return std::unexpected{ok.error()};
  }

  Building building;
  // No expectation points in the grammar, so nothing below throws; the catch is
  // what keeps that a property of this file rather than of every caller.
  const Silent quiet;
  const bool ok = [&]() noexcept {
    try {
      return bp::parse(source,
                       bp::with_error_handler(bp::with_globals(subscript, building), quiet),
                       bp::ws);
    } catch (...) {
      return false;
    }
  }();

  if (!ok) {
    return fail(building.why);
  }
  return impl::finish_subscripts(building.subs);
}

result<Plan> plan(const std::string_view source) noexcept {
  return parse(source).and_then(impl::make_plan);
}

} // namespace einsum::rt
