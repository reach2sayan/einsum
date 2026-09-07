// Under BOOST_NO_EXCEPTIONS boost::throw_exception is declared but not defined,
// and the program must supply it.  Nothing in einsum routes through it -- the
// error channel is std::expected and every container push here is bound-checked
// first -- only Boost's own internals do, where the alternative to terminating
// is a container in a state Boost will not describe.
//
// Default visibility explicitly: libeinsum_rt is -fvisibility=hidden and
// Boost's inline code in other objects has to find these.  They live in
// namespace boost, so EINSUM_API is not theirs to wear.
#include <boost/assert/source_location.hpp>
#include <boost/throw_exception.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace boost {

#if defined(__GNUC__) || defined(__clang__)
#define EINSUM_BOOST_THROW_VISIBILITY __attribute__((visibility("default")))
#else
#define EINSUM_BOOST_THROW_VISIBILITY
#endif

EINSUM_BOOST_THROW_VISIBILITY void throw_exception(const std::exception &e) {
  std::fprintf(stderr, "einsum: boost threw with exceptions disabled: %s\n", e.what());
  std::abort();
}

EINSUM_BOOST_THROW_VISIBILITY void throw_exception(const std::exception &e,
                                                   const source_location &loc) {
  std::fprintf(stderr, "einsum: boost threw with exceptions disabled: %s (%s:%u)\n",
               e.what(), loc.file_name(), loc.line());
  std::abort();
}

} // namespace boost
