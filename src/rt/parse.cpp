#include "einsum/rt/parse.hpp"

#include "einsum/core/einsum_object.hpp"
#include "einsum/core/limits.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/error.hpp"

#include <boost/parser/parser.hpp>

#include <cstdint>
#include <string_view>

// The subscript grammar, and the only translation unit that sees Boost.Parser
// -- which wraps every parse() in a try/catch and so cannot compile under the
// -fno-exceptions the rest of einsum is built with.  The CMakeLists beside this
// file takes the flag off this one object, and the catch below keeps that
// local: nothing thrown here reaches a caller.
//
// Everything this file decides that is not the shape of the text it decides by
// calling the same helpers impl::parse_subscripts does, so the two front ends
// cannot drift apart; tests/tests_parse.cpp checks that they have not.
namespace einsum {
namespace {

namespace bp = boost::parser;

// std::tuple, never boost::hana::tuple: nothing in this project may pull Hana
// in, and a default that flipped would do so silently.
static_assert(BOOST_PARSER_USE_STD_TUPLE,
              "einsum::rt: Boost.Parser must use std::tuple, not Hana");

using bp::_attr;
using bp::_globals;
using bp::_pass;

// Boost.Parser's own handler writes a caret diagnostic to a stream.  einsum
// reports on the numeric path alone, so this one says only that the parse
// failed and leaves the errc to say what about it did.
struct Silent {
  template <typename Iter, typename Sentinel>
  constexpr bp::error_handler_result
  operator()(Iter, Sentinel, const bp::parse_error<Iter> &) const {
    return bp::error_handler_result::fail;
  }
  template <typename Context>
  constexpr void diagnose(bp::diagnostic_kind, std::string_view,
                          const Context &) const {}
  template <typename Context, typename Iter>
  constexpr void diagnose(bp::diagnostic_kind, std::string_view,
                          const Context &, Iter) const {}
};

// Carried through with_globals: BOOST_PARSER_DEFINE_RULES wants its rules at
// namespace scope, and a synthesized attribute would not do either --
// `(operand % ',') >> -("->" >> operand)` collapses the output into the
// operand vector.
struct Building {
  Subscripts subs{};
  Labels current{};
  std::uint8_t current_ellipsis = kNoEllipsis;
  bool in_output = false;
  errc why = errc::bad_syntax; // what the refusing action meant, if one did
  bool refused = false;
};

namespace act {

// Name the code, fail the parser, and let the caller read `why` back.  The
// first refusal wins: a failing parser backtracks, and the alternative's own
// complaint would bury the one that mattered.
template <errc C>
constexpr auto refuse = [](auto &ctx) {
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
  if (!target.try_push_back(_attr(ctx))) {
    refuse<errc::rank_too_high>(ctx);
  }
};

constexpr auto end_operand = [](auto &ctx) {
  Building &building = _globals(ctx);
  // Stored together, so the two vectors cannot fall out of step.
  if (!building.subs.operands.try_push_back(building.current) ||
      !building.subs.ellipsis_at.try_push_back(building.current_ellipsis)) {
    refuse<errc::too_many_operands>(ctx);
    return;
  }
  building.current.clear();
  building.current_ellipsis = kNoEllipsis;
};

// At most one per term, at whatever position it was written.
constexpr auto mark_ellipsis = [](auto &ctx) {
  Building &building = _globals(ctx);
  std::uint8_t &at = building.in_output ? building.subs.output_ellipsis_at
                                        : building.current_ellipsis;
  if (at != kNoEllipsis) {
    refuse<errc::ellipsis_repeated>(ctx);
    return;
  }
  at =
      static_cast<std::uint8_t>(building.in_output ? building.subs.output.size()
                                                   : building.current.size());
};

// Fires on the arrow itself, so the labels after it land in the output.
constexpr auto begin_output = [](auto &ctx) {
  Building &building = _globals(ctx);
  building.in_output = true;
  building.subs.explicit_output = true;
};

} // namespace act

// Rules carry no attribute: the actions write into the globals.
bp::rule<struct label_tag> const label = "label";
bp::rule<struct operand_tag> const operand = "operand";
bp::rule<struct ellipsis_tag> const ellipsis = "ellipsis";
bp::rule<struct inputs_tag> const inputs = "operands";
bp::rule<struct output_tag> const output = "output";
bp::rule<struct subscript_tag> const subscript = "subscript";

constexpr auto letter = bp::char_('a', 'z') | bp::char_('A', 'Z');

auto const label_def = letter[act::push_label];
// Labels and at most one '...', in any order, each keeping its position.
auto const ellipsis_def = bp::lit("...")[act::mark_ellipsis];
auto const operand_def = +(label | ellipsis);
// eps: an empty operand is a different complaint from a bad subscript.
auto const inputs_def =
    (operand[act::end_operand] | bp::eps[act::refuse<errc::empty_operand>]) %
    ',';
auto const output_def = bp::lit("->")[act::begin_output] >> *(label | ellipsis);
auto const subscript_def = inputs >> -output >> bp::eoi;

BOOST_PARSER_DEFINE_RULES(label, operand, ellipsis, inputs, output, subscript);

} // namespace

result<Subscripts> parse_subscript(const std::string_view source) noexcept {
  if (const auto ok = impl::precheck(source); !ok) {
    return std::unexpected{ok.error()};
  }

  Building building;
  // No expectation points, so nothing below throws; the catch keeps that a
  // property of this file rather than of every caller.
  const Silent quiet;
  const bool ok = [&]() noexcept {
    try {
      return bp::parse(
          source,
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

result<Einsum> einsum(const std::string_view source, const path order) {
  return parse_subscript(source)
      .and_then(impl::make_plan)
      .transform([order](Plan &&plan) {
        return impl::einsum_access::make<impl::HeapScratch>(std::move(plan),
                                                            order);
      });
}

} // namespace einsum
