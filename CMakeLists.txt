cmake_minimum_required(VERSION 3.26)

# Newest GCC on PATH, unless the caller chose a compiler.  Must precede
# project(): the default g++ here is 13, which has no std::ranges::to.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
        AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(EINSUM_PREFERRED_CXX NAMES g++-15 g++-14)
    if (EINSUM_PREFERRED_CXX)
        set(CMAKE_CXX_COMPILER "${EINSUM_PREFERRED_CXX}")
    endif ()
endif ()

project(EinsteinSummation VERSION 2.0.0 LANGUAGES CXX)

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
option(EINSUM_BUILD_SHARED_RT "Build libeinsum_rt as a shared library" OFF)

# --- the header-only library -------------------------------------------------
einsum_use_boost()
einsum_use_eigen()
einsum_use_mdspan()

add_library(einsum INTERFACE)
add_library(einsum::einsum ALIAS einsum)

target_sources(einsum INTERFACE
        FILE_SET HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES
        include/einsum/einsum.hpp
        include/einsum/util/concepts.hpp
        include/einsum/util/error.hpp
        include/einsum/util/export.hpp
        include/einsum/util/fixed_string.hpp
        include/einsum/util/fixed_vec.hpp
        include/einsum/util/ranges.hpp
        include/einsum/core/einsum_object.hpp
        include/einsum/core/execute.hpp
        include/einsum/core/kernels.hpp
        include/einsum/core/limits.hpp
        include/einsum/core/lower.hpp
        include/einsum/core/owned.hpp
        include/einsum/core/path.hpp
        include/einsum/core/plan.hpp
        include/einsum/core/view.hpp
        include/einsum/ct/einsum.hpp
        include/einsum/ct/static_shape.hpp
        include/einsum/ct/subscripts.hpp
        include/einsum/rt/parse.hpp)

target_include_directories(einsum INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(einsum INTERFACE Boost::headers Eigen3::Eigen mdspan::mdspan)
target_compile_features(einsum INTERFACE cxx_std_23)

# --- libeinsum_rt ------------------------------------------------------------
add_subdirectory(src/rt)

# A consumer that add_subdirectory()s us gets the library and nothing else.
if (NOT PROJECT_IS_TOP_LEVEL)
    return ()
endif ()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE einsum::einsum einsum::rt)
einsum_target_flags(main)
einsum_target_pch(main)
einsum_target_trace(main)

enable_testing()
if (EINSUM_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

if (EINSUM_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif ()
