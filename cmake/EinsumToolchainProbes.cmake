# The floor the toolchain must clear.  <expected> is the gate: libstdc++ hides it
# behind __cpp_concepts, which clang++-18 does not define at the level libstdc++
# asks for, so the header is present and empty.  std::ranges::to needs GCC 14.
# Neither shows up as a version check -- only as a compile.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS "-std=c++23")
check_cxx_source_compiles(
        "#include <expected>
         #include <ranges>
         #include <vector>
         int main() {
           auto v = std::views::iota(0, 3) | std::ranges::to<std::vector<int>>();
           return std::expected<int, int>{static_cast<int>(v.size())}.value() - 3;
         }"
        EINSUM_TOOLCHAIN_OK)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT MSVC AND NOT EINSUM_TOOLCHAIN_OK)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} cannot build "
            "EinsteinSummation, which needs <expected> and std::ranges::to.  Build with "
            "GCC 14+ or Clang 20+.  Clang 18 is out because libstdc++ gates <expected> "
            "on __cpp_concepts, which that release does not define high enough -- the "
            "header is there and declares nothing.  The compiler's own diagnostic is in "
            "the CMakeConfigureLog.")
endif ()
