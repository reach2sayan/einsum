# The flags our own targets are built with.  Nothing here touches the INTERFACE
# library: a flag reaches a target only through einsum_target_flags().
include_guard(GLOBAL)

option(EINSUM_NATIVE_ARCH "Build optimized for this machine" ON)
option(EINSUM_TRACE "Emit compile-time profiling info (Clang: -ftime-trace, GCC: -ftime-report)" OFF)

# --- ccache ------------------------------------------------------------------
# Off under EINSUM_TRACE: a cache hit restores the .o and never writes the
# -ftime-trace .json beside it, so the trace silently comes back empty.
if (NOT EINSUM_TRACE)
    find_program(EINSUM_CCACHE ccache)
    if (EINSUM_CCACHE)
        message(STATUS "ccache: ${EINSUM_CCACHE}")
        # include() runs in the caller's scope, so a plain set() is the right one.
        set(CMAKE_C_COMPILER_LAUNCHER "${EINSUM_CCACHE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${EINSUM_CCACHE}")
    endif ()
else ()
    message(STATUS "EINSUM_TRACE=ON: ccache disabled so the per-TU traces are emitted")
endif ()

# --- flags -------------------------------------------------------------------
if (MSVC)
    # /Zc:preprocessor: util/error.hpp builds the errc table with
    # Boost.Preprocessor, which the traditional preprocessor only reaches
    # through its own workaround headers.  /constexpr:* because the whole
    # planner runs at compile time and MSVC's evaluator has the lowest default
    # budget of the three.  /external:anglebrackets is safe here because our own
    # headers are quoted without exception -- only Boost, Eigen, mdspan and the
    # standard library arrive in angle brackets.
    set(EINSUM_CODEGEN_FLAGS
            /arch:AVX2 /bigobj
            /Zc:preprocessor /Zc:__cplusplus /utf-8
            /constexpr:steps10000000 /constexpr:depth2048
            /EHs-c- /D_HAS_EXCEPTIONS=0 /wd4577)
    set(EINSUM_WARNINGS /W4 /external:anglebrackets /external:W0)
    # einsum_rt is SHARED and every symbol it exports is hidden by default on
    # this platform too -- but here EINSUM_API covers only rt/parse.hpp's two
    # functions, and Boost's inline code in a consumer expects to find the
    # boost::throw_exception replacements as well.  Rather than decorate those
    # by hand in namespace boost, let the linker export what the objects define.
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else ()
    set(EINSUM_CODEGEN_FLAGS "")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        if (EINSUM_NATIVE_ARCH)
            set(EINSUM_CODEGEN_FLAGS -march=native)
        else ()
            set(EINSUM_CODEGEN_FLAGS -march=x86-64-v3)
        endif ()
    endif ()
    # -funwind-tables: -fno-exceptions still has to be walkable by a profiler.
    list(APPEND EINSUM_CODEGEN_FLAGS -fno-exceptions -funwind-tables)
    list(APPEND EINSUM_CODEGEN_FLAGS
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2>
            $<$<CONFIG:RelWithDebInfo>:-g>)
    set(EINSUM_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
endif ()

# Global: a sanitizer has to instrument everything, Boost.Test included.
set(EINSUM_SANITIZE "off" CACHE STRING "off | address | undefined | thread")
set_property(CACHE EINSUM_SANITIZE PROPERTY STRINGS off address undefined thread)
if (NOT EINSUM_SANITIZE STREQUAL "off")
    if (MSVC)
        message(WARNING "EINSUM_SANITIZE=${EINSUM_SANITIZE} ignored: MSVC has no such sanitizer")
    else ()
        add_compile_options(-fsanitize=${EINSUM_SANITIZE} -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=${EINSUM_SANITIZE})
    endif ()
endif ()

# einsum_target_flags(<target> [EXCEPTIONS])
# EXCEPTIONS keeps exceptions on for one target: Boost.Parser's parse() is a
# try/catch, and Boost.Test's whole reporting path is throw-based.
function(einsum_target_flags target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "EXCEPTIONS" "" "")
    set(flags ${EINSUM_CODEGEN_FLAGS})
    if (arg_EXCEPTIONS)
        list(REMOVE_ITEM flags -fno-exceptions /EHs-c- /D_HAS_EXCEPTIONS=0)
    endif ()
    target_compile_options(${target} PRIVATE ${flags} ${EINSUM_WARNINGS})
    # Not on MSVC: it reports command-line warnings no source change can
    # silence -- D9025 for the /EHsc the generator adds and this function then
    # overrides with /EHs-c- -- and a build cannot fail on those.
    if (NOT MSVC)
        set_property(TARGET ${target} PROPERTY COMPILE_WARNING_AS_ERROR ON)
    endif ()
endfunction()

# einsum_runtime_deps(<target>)
# Everything that links einsum::rt has to find libeinsum_rt at run time.  On
# ELF that is one RPATH entry; on Windows there is no such thing, and the
# loader looks beside the binary -- so the DLLs the target links get copied
# there after it is built.  $<TARGET_RUNTIME_DLLS> is empty on every other
# platform, and the whole command is skipped there anyway.
function(einsum_runtime_deps target)
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "$ORIGIN")
    if (WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
                COMMAND_EXPAND_LISTS)
    endif ()
endfunction()

# --- compile-time profiling --------------------------------------------------
# Clang: -ftime-trace writes <object>.json beside the object, for
# chrome://tracing or speedscope.app.  GCC: -ftime-report goes to stderr.
function(einsum_target_trace target)
    if (NOT EINSUM_TRACE)
        return ()
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE -ftime-trace)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${target} PRIVATE -ftime-report)
    endif ()
endfunction()

# --- precompiled headers -----------------------------------------------------
# Per target rather than shared, because a PCH is only reusable across objects
# built with the same flags -- and the EXCEPTIONS targets are not.
set(EINSUM_PCH_HEADERS
        <algorithm>
        <array>
        <concepts>
        <cstddef>
        <expected>
        <ranges>
        <span>
        <string_view>
        <type_traits>
        <Eigen/Core>
        <experimental/mdspan>)

function(einsum_target_pch target)
    target_precompile_headers(${target} PRIVATE ${EINSUM_PCH_HEADERS})
endfunction()
