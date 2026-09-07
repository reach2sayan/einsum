# The floor the toolchain must clear.  <expected> is the gate: libstdc++ hides it
# behind __cpp_concepts, which clang++-18 does not define at the level libstdc++
# asks for, so the header is present and empty.  That does not show up as a
# version check -- only as a compile.
#
# The probe asks for <expected> and nothing else, because <expected> is the only
# library feature the headers cannot be written without.  It used to ask for
# std::ranges::to as well, which no longer appears anywhere in einsum and which
# libstdc++ only provides from 14: that turned a clang 20 paired with an older
# libstdc++ into a refusal to configure over a feature the code never uses.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS "-std=c++23")
check_cxx_source_compiles(
        "#include <expected>
         int main() {
           return std::expected<int, int>{3}.value() - 3;
         }"
        EINSUM_TOOLCHAIN_OK)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT MSVC AND NOT EINSUM_TOOLCHAIN_OK)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} cannot build "
            "EinsteinSummation, which needs a working <expected>.  Build with GCC 14+ "
            "or Clang 19+.  Clang 18 is out because libstdc++ gates <expected> on "
            "__cpp_concepts, which that release does not define high enough -- the "
            "header is there and declares nothing.  The compiler's own diagnostic is "
            "in the CMakeConfigureLog.")
endif ()
