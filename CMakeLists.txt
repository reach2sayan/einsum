cmake_minimum_required(VERSION 3.26)

# Newest GCC on PATH, unless the caller chose a compiler.  Must precede
# project(): the default g++ here is 13, whose libstdc++ has no
# std::views::enumerate and no std::from_range, so it is below the floor the
# toolchain probe checks for.
# CMAKE_HOST_WIN32, not WIN32: this runs before project(), which is where WIN32
# is decided.  On Windows the compiler is whatever vcvars put on PATH, and a
# stray MinGW g++ is not what this preference is for.
if (NOT CMAKE_HOST_WIN32
        AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
        AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(EINSUM_PREFERRED_CXX NAMES g++-15 g++-14)
    if (EINSUM_PREFERRED_CXX)
        set(CMAKE_CXX_COMPILER "${EINSUM_PREFERRED_CXX}")
    endif ()
endif ()

project(einsum VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
            STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(EinsumToolchainProbes)
include(EinsumCompileOptions)
include(EinsumDependencies)

option(EINSUM_BUILD_TESTS "Build the Boost.Test suites" ON)
option(EINSUM_BUILD_BENCHMARKS "Build the Google Benchmark suite" ON)
option(EINSUM_BUILD_PYTHON "Build the pybind11 extension module" OFF)
# Off here, on in the Linux wheel (pyproject.toml) and the python preset.  A
# library that spawns threads is a different thing from one that does not, so
# a plain configure gets the one it always had.  See einsum_use_openmp.
option(EINSUM_OPENMP "Run Eigen's GEMM on an OpenMP team" OFF)

# --- the header-only library -------------------------------------------------
einsum_use_boost()
einsum_use_eigen()
einsum_use_mdspan()

add_library(einsum INTERFACE)
add_library(einsum::einsum ALIAS einsum)
if (EINSUM_OPENMP)
    einsum_use_openmp()
    target_link_libraries(einsum INTERFACE OpenMP::OpenMP_CXX)
endif ()

# Every header under include/, found rather than listed: the list drifted once
# already -- core/kind.hpp was written, not added, and only an install noticed.
file(GLOB_RECURSE EINSUM_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/einsum/*.hpp")
target_sources(einsum INTERFACE
        FILE_SET HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES ${EINSUM_HEADERS})

target_include_directories(einsum INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
# BUILD_INTERFACE only: an installed einsum carries Boost, Eigen and mdspan in
# its own include directory (cmake/EinsumInstall.cmake), so the one path above
# is all three and the exported target names no package a consumer must find.
target_link_libraries(einsum INTERFACE
        $<BUILD_INTERFACE:Boost::headers>
        $<BUILD_INTERFACE:Eigen3::Eigen>
        $<BUILD_INTERFACE:mdspan::mdspan>)
# The one thing those targets carry that is not an include path.
target_compile_definitions(einsum INTERFACE $<INSTALL_INTERFACE:EIGEN_MPL2_ONLY>)
target_compile_features(einsum INTERFACE cxx_std_23)

# The flags in EinsumCompileOptions.cmake are attached PRIVATE, which is right
# for the ones that decide how *this* project is compiled and wrong for the few
# that decide whether a consumer compiles at all.  einsum is headers but for one
# translation unit: the whole compile-time planner behind einsum("ij,jk->ik"_ct)
# is constant evaluation in the consumer's own TU, against MSVC's evaluator --
# the one with the lowest default budget of the three -- and util/error.hpp
# builds the errc table with Boost.Preprocessor SEQ macros, which the
# traditional preprocessor reaches only through workaround headers.  So MSVC
# needs these on the far side of the interface, and contrib/nuget/einsum.targets
# restates the same list for a consumer arriving through MSBuild instead.  Keep
# the two in step.
#
# Not here, deliberately: /arch:AVX2 is einsum_rt's own codegen and putting it
# on a consumer would raise their CPU floor silently; /EHs-c- and
# _HAS_EXCEPTIONS=0 are this project's choice and not one to impose -- a
# consumer compiling with exceptions gets a Boost that throws inline in its own
# TU rather than calling the boost::throw_exception einsum_rt supplies, which is
# self-consistent either way; /W4 and the /external:* pair are their warning
# policy, not ours.
target_compile_options(einsum INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:
        /bigobj /utf-8 /Zc:preprocessor /Zc:__cplusplus
        /constexpr:steps10000000 /constexpr:depth2048>)

# --- the vendored header list ------------------------------------------------
# What an installed einsum ships is decided by scripts/vendor_headers.py, and a
# public header naming a library it did not before leaves that list stale --
# which CI catches, but only after a push.  So the build checks it too.
#
# Checked, never rewritten: the list is the compiler's include closure on one
# machine, and a build that regenerated it would make a tracked file depend on
# who built last.  The stamp is what keeps this off the hot path -- the script
# runs a compiler, so it reruns only when a public header, the pins, the script
# or the list itself changes.
#
# Off on Windows whatever else is true: scripts/vendor_headers.py asks the
# compiler for the closure with `-std=c++23 ... -M`, which is a GCC/Clang
# spelling cl.exe does not have.  The list stays a Linux-generated artifact --
# it already carries the MSVC arms of boost/config and boost/preprocessor,
# because those three trees are vendored whole -- so an MSVC build consumes it
# happily and simply does not re-derive it.
if (WIN32)
    set(EINSUM_CHECK_VENDORED_HEADERS_DEFAULT OFF)
else ()
    set(EINSUM_CHECK_VENDORED_HEADERS_DEFAULT ${PROJECT_IS_TOP_LEVEL})
endif ()
option(EINSUM_CHECK_VENDORED_HEADERS
        "Fail the build when cmake/EinsumVendoredHeaders.cmake is stale"
        ${EINSUM_CHECK_VENDORED_HEADERS_DEFAULT})
if (EINSUM_CHECK_VENDORED_HEADERS)
    find_package(Python 3.11 QUIET COMPONENTS Interpreter)
    if (NOT Python_Interpreter_FOUND)
        message(STATUS "einsum: no Python 3.11+; not checking the vendored "
                "header list")
    else ()
        set(EINSUM_VENDORED_STAMP
                "${CMAKE_CURRENT_BINARY_DIR}/vendored_headers.stamp")
        add_custom_command(OUTPUT "${EINSUM_VENDORED_STAMP}"
                COMMAND "${Python_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/vendor_headers.py" --check
                COMMAND "${CMAKE_COMMAND}" -E touch "${EINSUM_VENDORED_STAMP}"
                DEPENDS ${EINSUM_HEADERS}
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/vendor_headers.py"
                "${CMAKE_CURRENT_LIST_DIR}/cmake/EinsumVendoredHeaders.cmake"
                "${CMAKE_CURRENT_LIST_DIR}/cmake/EinsumDependencies.cmake"
                COMMENT "Checking the vendored header list"
                VERBATIM)
        add_custom_target(einsum_vendored_headers ALL
                DEPENDS "${EINSUM_VENDORED_STAMP}")
    endif ()
endif ()

# --- libeinsum_rt ------------------------------------------------------------
add_subdirectory(src/rt)

option(EINSUM_INSTALL "Generate the install rules and the CMake package"
        ${PROJECT_IS_TOP_LEVEL})
if (EINSUM_INSTALL)
    include(EinsumInstall)
endif ()

# A consumer that add_subdirectory()s us gets the library and nothing else.
if (NOT PROJECT_IS_TOP_LEVEL)
    return ()
endif ()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE einsum::einsum einsum::rt)
einsum_target_flags(main)
einsum_target_pch(main)
einsum_target_trace(main)
einsum_runtime_deps(main)

enable_testing()
if (EINSUM_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

if (EINSUM_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif ()

# After enable_testing(): python/ registers the pytest suite with it, so `ctest`
# stays the one verb for "run the tests".
if (EINSUM_BUILD_PYTHON)
    einsum_use_pybind11()
    add_subdirectory(python)
endif ()
