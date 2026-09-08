cmake_minimum_required(VERSION 3.26)

# Newest GCC on PATH, unless the caller chose a compiler.  Must precede
# project(): the default g++ here is 13, whose libstdc++ has no
# std::views::enumerate and no std::from_range, so it is below the floor the
# toolchain probe checks for.
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
option(EINSUM_BUILD_PYTHON "Build the pybind11 extension module" OFF)

# --- the header-only library -------------------------------------------------
einsum_use_boost()
einsum_use_eigen()
einsum_use_mdspan()

add_library(einsum INTERFACE)
add_library(einsum::einsum ALIAS einsum)

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
