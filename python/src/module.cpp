#include "array.hpp"
#include "error.hpp"

#include "einsum/core/limits.hpp"
#include "einsum/core/path.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/rt/dynamic.hpp"
#include "einsum/rt/parse.hpp"
#include "einsum/util/error.hpp"

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <boost/container/static_vector.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include <exception>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

namespace einsum::py {
namespace {

// Deliberately leaked: it lives as long as the module, and a translator is a
// plain function pointer that cannot capture.
pyb::handle error_class;

// Back as the string it was parsed from, which is what Einsum.subscripts has to
// answer.  Nothing in C++ asks, so the library has no formatter for one.
[[nodiscard]] std::string render(const Subscripts &subs) {
  const auto term = [](const Labels &labels, const std::uint8_t ellipsis_at) {
    std::string out;
    for (const auto [i, label] : labels | std::views::enumerate) {
      if (static_cast<std::uint8_t>(i) == ellipsis_at) {
        out += "...";
      }
      out += label;
    }
    // At any position within the term, the end included.
    if (ellipsis_at == labels.size()) {
      out += "...";
    }
    return out;
  };

  std::string out;
  for (const auto [i, labels] : subs.operands | std::views::enumerate) {
    if (i != 0) {
      out += ',';
    }
    out += term(labels, subs.ellipsis_at[static_cast<std::size_t>(i)]);
  }
  if (subs.explicit_output) {
    out += "->";
    out += term(subs.output, subs.output_ellipsis_at);
  }
  return out;
}

// One call in one scalar, written straight into the result.
template <typename T>
[[nodiscard]] pyb::object contract(const Einsum &self, const pyb::args &operands) {
  if (operands.size() > kMaxOperands) {
    fail_with(errc::too_many_operands);
  }
  boost::container::static_vector<Operand<T>, kMaxOperands> held;
  boost::container::static_vector<impl::TensorView<const T>, kMaxOperands> views;
  for (const pyb::handle operand : operands) {
    held.push_back(as_operand<T>(operand));
    views.push_back(held.back().view);
  }

  pyb::array_t<T> out;
  unwrap(impl::evaluate_dynamic<T>(
      self, std::span<const impl::TensorView<const T>>{views},
      [&out](const Shape &shape) {
        out = empty_like<T>(shape);
        return out.mutable_data();
      }));
  return std::move(out);
}

} // namespace
} // namespace einsum::py

PYBIND11_MODULE(_einsum, m) {
  namespace ei = einsum;
  namespace ep = einsum::py;
  namespace pyb = pybind11;

  m.doc() = "einsum's runtime path: a subscript parsed at run time, and the "
            "object it answers.  Import einsum instead.";
  m.attr("__version__") = EINSUM_VERSION_STRING;

  // What a plan costs whether or not a subscript uses them.
  m.attr("MAX_RANK") = ei::kMaxRank;
  m.attr("MAX_OPERANDS") = ei::kMaxOperands;

  // A translator, not register_exception: the code goes on the instance.
  ep::error_class =
      pyb::exception<ep::PyError>(m, "Error", PyExc_ValueError).release();
  pyb::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const ep::PyError &e) {
      pyb::object raised = ep::error_class(e.what());
      raised.attr("code") = pyb::cast(e.code);
      PyErr_SetObject(ep::error_class.ptr(), raised.ptr());
    }
  });

  // From the one table, so a code added to EINSUM_ERRC_SEQ appears here.
  pyb::native_enum<ei::errc> errors(m, "errc", "enum.IntEnum",
                                    "Why einsum refused; Error.code carries one.");
#define EINSUM_PY_ERRC(r, unused, elem)                                        \
  errors.value(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(0, elem)),               \
               ei::errc::BOOST_PP_TUPLE_ELEM(0, elem),                         \
               BOOST_PP_TUPLE_ELEM(1, elem));
  BOOST_PP_SEQ_FOR_EACH(EINSUM_PY_ERRC, ~, EINSUM_ERRC_SEQ)
#undef EINSUM_PY_ERRC
  errors.finalize();

  pyb::native_enum<ei::path>(m, "Path", "enum.IntEnum",
                             "In what order the operands are contracted.")
      .value("GREEDY", ei::path::greedy,
             "Cheapest pair first, by a greedy search over the pairs.")
      .value("SEQUENTIAL", ei::path::sequential,
             "The subscript's own order, left to right, verbatim.")
      .finalize();

  // Registered before the functions answering one, or pybind11 writes the raw
  // C++ spelling into their signatures.  No init: einsum() is the only way to
  // one, and it is where a bad subscript is refused.
  pyb::class_<ei::Einsum>(m, "Einsum",
                          "A parsed subscript.  Call it with arrays.")
      // The GIL is held throughout, deliberately: a call rewrites the object's
      // scratch and lowering cache, so two threads in one Einsum would race.
      // A caller wanting parallelism copies the object.
      .def(
          "__call__",
          [](const ei::Einsum &self, const pyb::args &operands) {
            return ep::promote(operands) == ep::dtype::f32
                       ? ep::contract<float>(self, operands)
                       : ep::contract<double>(self, operands);
          },
          pyb::doc("Contract the operands and hand back the result as an "
                   "ndarray, at the rank the subscript implies."))
      .def_property_readonly(
          "subscripts",
          [](const ei::Einsum &self) { return ep::render(self.subscripts()); },
          pyb::doc("The subscript this was built from."))
      .def_property_readonly("operand_count", &ei::Einsum::operand_count,
                             pyb::doc("How many operands the subscript names."))
      .def_property_readonly(
          "output_labels",
          [](const ei::Einsum &self) {
            const ei::Labels &labels = self.output_labels();
            return std::string{labels.begin(), labels.end()};
          },
          pyb::doc("The output's labels, in the order the result is indexed."))
      .def("__repr__", [](const ei::Einsum &self) {
        return "Einsum('" + ep::render(self.subscripts()) + "')";
      });

  m.def(
      "parse_subscript",
      [](const std::string_view source) {
        return ep::render(ep::unwrap(ei::parse_subscript(source)));
      },
      pyb::arg("subscripts"),
      pyb::doc("The subscript, parsed and written back out.  Raises Error with "
               "the code the grammar refused with."));

  m.def(
      "einsum",
      [](const std::string_view source, const ei::path order) {
        return ep::unwrap(ei::einsum(source, order));
      },
      pyb::arg("subscripts"), pyb::kw_only(),
      pyb::arg_v("path", ei::path::greedy, "Path.GREEDY"),
      pyb::doc("Parse a subscript and hand back the object that runs it."));

}
