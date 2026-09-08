# The floor the toolchain must clear, expressed as a compile rather than a
# version check, because neither half of it shows up as one.
#
# <expected> is the first gate: libstdc++ hides it behind __cpp_concepts, which
# clang++-18 does not define at the level libstdc++ asks for, so the header is
# present and declares nothing.
#
# The rest is the libstdc++ 14 floor.  It is probed with std::views::enumerate
# and std::from_range because those are what einsum actually uses -- enumerate
# in ct/subscripts.hpp and core/einsum_object.hpp, from_range in
# util/fixed_vec.hpp and core/kernels.hpp.  This probe used to ask for
# std::ranges::to instead, which stands at the same libstdc++ level but appears
# nowhere in the library: a compiler could then be refused over a feature the
# build would never have reached, and -- worse -- the day ranges::to moved
# without enumerate moving with it, the gate would have been measuring the
# wrong thing.  Probe what the code needs.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

# No CMAKE_REQUIRED_FLAGS: try_compile() carries CMAKE_CXX_STANDARD into the
# probe, and the root CMakeLists sets it to 23 before including this file.  It
# used to say -std=c++23 by hand, which asks cl.exe for an option it does not
# have -- the probe then measured the default standard, and would have passed
# or failed for a reason unrelated to the floor.
check_cxx_source_compiles(
        "#include <expected>
         #include <ranges>
         #include <vector>
         struct Sink {
           int total = 0;
           constexpr Sink(std::from_range_t, const std::vector<int> &xs) {
             for (const auto [i, x] : xs | std::views::enumerate) {
               total += static_cast<int>(i) * x;
             }
           }
         };
         int main() {
           const std::vector<int> xs{1, 1, 1};
           const Sink s{std::from_range, xs};
           return std::expected<int, int>{s.total}.value() - 3;
         }"
        EINSUM_TOOLCHAIN_OK)

if (NOT EINSUM_TOOLCHAIN_OK)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} cannot build "
            "einsum, which needs <expected>, std::views::enumerate and "
            "std::from_range.  Build with GCC 14+, MSVC 19.38+ (Visual Studio 2022 "
            "17.8), or Clang 20+ against libstdc++ 14+ -- with clang it is the "
            "standard library that decides, not the compiler, so a new clang paired "
            "with an old libstdc++ lands here.  Clang 18 is out on <expected> "
            "regardless: libstdc++ gates it on __cpp_concepts, which that release "
            "does not define high enough, so the header is there and declares "
            "nothing.  The compiler's own diagnostic is in the CMakeConfigureLog.")
endif ()
