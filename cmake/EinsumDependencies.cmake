# Every third-party dependency, pinned and SHA-verified.  Nothing is ever
# find_package()d: a machine's Boost is whatever the distro froze, and this
# library needs Boost.Parser, which arrived in 1.87.
#
#   einsum_use_boost()             Boost, header-only, fetched   always
#   einsum_use_eigen()             Eigen 3.4.0, fetched          always
#   einsum_use_mdspan()            kokkos mdspan, fetched        always
#   einsum_use_googlebenchmark()   Google Benchmark, fetched     EINSUM_BUILD_BENCHMARKS
include_guard(GLOBAL)

include(FetchContent)

# --- where fetched sources land ---------------------------------------------
# Outside the build tree, so every preset shares one unpack and `rm -rf build`
# does not re-download 130 MB.
cmake_path(SET _einsum_deps_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../.deps")
set(EINSUM_DEPS_DIR "${_einsum_deps_default}" CACHE PATH
        "Where fetched sources are unpacked; shared by every build tree")
unset(_einsum_deps_default)

# --- versions ----------------------------------------------------------------
set(EINSUM_BOOST_VERSION "1.92.0" CACHE STRING "Boost release to fetch")
set(EINSUM_BOOST_SHA256 "ea7b982002cc9dfbe59b0b217b206f470dc75f3de0bb2973d844118934d82411"
        CACHE STRING "SHA256 of the archive EINSUM_BOOST_VERSION names")
set(EINSUM_EIGEN_VERSION "3.4.0" CACHE STRING "Eigen release to fetch")
set(EINSUM_EIGEN_SHA256 "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"
        CACHE STRING "SHA256 of the archive EINSUM_EIGEN_VERSION names")
# A commit, not a tag: mdspan's `stable` moves, and a moving pin is not one.
set(EINSUM_MDSPAN_COMMIT "80fc772eb812b45097c28fc0a46d8ff006138d69"
        CACHE STRING "kokkos/mdspan commit to fetch")
set(EINSUM_GOOGLEBENCHMARK_VERSION "1.9.1" CACHE STRING "Google Benchmark release to fetch")

# --- overrides ---------------------------------------------------------------
# Each names an include root and turns its fetch off entirely.
set(EINSUM_BOOST_INCLUDEDIR "" CACHE PATH
        "A Boost include root to use instead of the pinned fetch; must hold boost/version.hpp")
set(EINSUM_EIGEN_INCLUDEDIR "" CACHE PATH
        "An Eigen include root to use instead of the pinned fetch; must hold Eigen/Core")

# --- declarations ------------------------------------------------------------
# SOURCE_SUBDIR names nothing, so MakeAvailable unpacks and stops -- neither
# Boost nor Eigen is built, and neither one's CMake gets a say in our cache.
FetchContent_Declare(boost
        URL https://github.com/boostorg/boost/releases/download/boost-${EINSUM_BOOST_VERSION}/boost-${EINSUM_BOOST_VERSION}-b2-nodocs.tar.xz
        URL_HASH SHA256=${EINSUM_BOOST_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR einsum-does-not-build-boost
        SOURCE_DIR "${EINSUM_DEPS_DIR}/boost-${EINSUM_BOOST_VERSION}"
        SUBBUILD_DIR "${EINSUM_DEPS_DIR}/boost-${EINSUM_BOOST_VERSION}-subbuild"
        BINARY_DIR "${EINSUM_DEPS_DIR}/boost-${EINSUM_BOOST_VERSION}-build")

FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/${EINSUM_EIGEN_VERSION}/eigen-${EINSUM_EIGEN_VERSION}.tar.gz
        URL_HASH SHA256=${EINSUM_EIGEN_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR einsum-does-not-build-eigen
        SOURCE_DIR "${EINSUM_DEPS_DIR}/eigen-${EINSUM_EIGEN_VERSION}"
        SUBBUILD_DIR "${EINSUM_DEPS_DIR}/eigen-${EINSUM_EIGEN_VERSION}-subbuild"
        BINARY_DIR "${EINSUM_DEPS_DIR}/eigen-${EINSUM_EIGEN_VERSION}-build")

# mdspan does build: its CMake is what defines mdspan::mdspan.
FetchContent_Declare(mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG ${EINSUM_MDSPAN_COMMIT}
        GIT_SHALLOW FALSE
        SYSTEM
        SOURCE_DIR "${EINSUM_DEPS_DIR}/mdspan-${EINSUM_MDSPAN_COMMIT}"
        SUBBUILD_DIR "${EINSUM_DEPS_DIR}/mdspan-${EINSUM_MDSPAN_COMMIT}-subbuild")

FetchContent_Declare(googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v${EINSUM_GOOGLEBENCHMARK_VERSION}.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM)

# --- Boost -------------------------------------------------------------------
macro(einsum_use_boost)
    if (NOT TARGET Boost::headers)
        if (EINSUM_BOOST_INCLUDEDIR)
            _einsum_adopt_header_dep(Boost::headers "${EINSUM_BOOST_INCLUDEDIR}" "boost/version.hpp"
                    EINSUM_BOOST_INCLUDEDIR "Boost ${EINSUM_BOOST_VERSION}")
        else ()
            FetchContent_MakeAvailable(boost)
            _einsum_adopt_header_dep(Boost::headers "${boost_SOURCE_DIR}" "boost/version.hpp"
                    EINSUM_BOOST_INCLUDEDIR "Boost ${EINSUM_BOOST_VERSION}")
        endif ()
    endif ()
endmacro()

# --- Eigen -------------------------------------------------------------------
# EIGEN_MPL2_ONLY: the LGPL corner of Eigen 3.4 (some sparse solvers) is not
# linked and must not become linkable by accident.
macro(einsum_use_eigen)
    if (NOT TARGET Eigen3::Eigen)
        if (EINSUM_EIGEN_INCLUDEDIR)
            _einsum_adopt_header_dep(Eigen3::Eigen "${EINSUM_EIGEN_INCLUDEDIR}" "Eigen/Core"
                    EINSUM_EIGEN_INCLUDEDIR "Eigen ${EINSUM_EIGEN_VERSION}")
        else ()
            FetchContent_MakeAvailable(eigen)
            _einsum_adopt_header_dep(Eigen3::Eigen "${eigen_SOURCE_DIR}" "Eigen/Core"
                    EINSUM_EIGEN_INCLUDEDIR "Eigen ${EINSUM_EIGEN_VERSION}")
        endif ()
        set_property(TARGET Eigen3::Eigen APPEND PROPERTY
                INTERFACE_COMPILE_DEFINITIONS EIGEN_MPL2_ONLY)
    endif ()
endmacro()

# --- mdspan ------------------------------------------------------------------
# No libstdc++ on the floor ships <mdspan>, so the reference implementation is
# not a convenience here.
macro(einsum_use_mdspan)
    if (NOT TARGET mdspan::mdspan)
        set(MDSPAN_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
        set(MDSPAN_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(MDSPAN_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
        set(MDSPAN_ENABLE_COMP_EXT_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(mdspan)
    endif ()
endmacro()

# --- Google Benchmark --------------------------------------------------------
macro(einsum_use_googlebenchmark)
    if (NOT TARGET benchmark::benchmark)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)
        _einsum_silence_dependency(benchmark benchmark_main)
    endif ()
endmacro()

# --- helpers -----------------------------------------------------------------
# One IMPORTED GLOBAL INTERFACE target per header-only root.  SYSTEM, so
# -Wconversion and friends never fire inside somebody else's headers.
macro(_einsum_adopt_header_dep target root witness override_var label)
    if (NOT EXISTS "${root}/${witness}")
        message(FATAL_ERROR
                "No ${label} at ${root}: it has no ${witness}.  ${override_var} must name "
                "the directory *containing* that path, and setting it turns the pinned "
                "fetch off entirely.  Unset it to build against the fetched copy instead.")
    endif ()
    add_library(${target} INTERFACE IMPORTED GLOBAL)
    set_target_properties(${target} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${root}"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${root}")
    message(STATUS "${label} (${root})")
endmacro()

function(_einsum_silence_dependency)
    if (NOT MSVC)
        return ()
    endif ()
    foreach (target IN LISTS ARGV)
        if (TARGET ${target})
            target_compile_options(${target} PRIVATE /W0)
            set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
        endif ()
    endforeach ()
endfunction()
