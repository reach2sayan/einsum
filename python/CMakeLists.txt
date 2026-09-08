# The extension module, a consumer of libeinsum_rt.  The wheel is these two
# files:
#   einsum/_einsum.cpython-*.so   these bindings
#   einsum/libeinsum_rt.so.2      the library they link

pybind11_add_module(einsum_python src/module.cpp)
set_target_properties(einsum_python PROPERTIES OUTPUT_NAME _einsum)
target_link_libraries(einsum_python PRIVATE einsum::einsum einsum::rt)
target_compile_definitions(einsum_python PRIVATE
        EINSUM_VERSION_STRING="${PROJECT_VERSION}")

# EXCEPTIONS: pybind11's error_already_set is a C++ exception, and every
# conversion in this TU can raise one.  No PCH: the shared one is built
# -fno-exceptions and would not be reusable here.
einsum_target_flags(einsum_python EXCEPTIONS)
einsum_runtime_deps(einsum_python)

set_target_properties(einsum_python PROPERTIES
        INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>"
        BUILD_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>")
install(TARGETS einsum_python LIBRARY DESTINATION einsum)
install(TARGETS einsum_rt
        LIBRARY DESTINATION einsum NAMELINK_SKIP
        RUNTIME DESTINATION einsum
        ARCHIVE DESTINATION einsum EXCLUDE_FROM_ALL)
# A wheel holds no symlinks: the real file takes the soname.
if (NOT WIN32)
    install(CODE "file(RENAME
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/einsum/$<TARGET_FILE_NAME:einsum_rt>\"
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/einsum/$<TARGET_SONAME_FILE_NAME:einsum_rt>\")")
endif ()

# --- the package, in the build tree ------------------------------------------
# $<1:...> keeps a multi-config generator from appending Release/.
set(EINSUM_PYTHON_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/einsum")
set_target_properties(einsum_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "$<1:${EINSUM_PYTHON_PACKAGE}>")
# Beside it, so PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR} is an importable package
# with no wheel and no install step.  Both directories, because which one a
# shared library lands in is the platform's choice: the .so is a LIBRARY
# artifact and the .dll is a RUNTIME one.  (_einsum itself needs only the first:
# a MODULE library is a LIBRARY artifact everywhere, Windows included.)
set_target_properties(einsum_rt PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "$<1:${EINSUM_PYTHON_PACKAGE}>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${EINSUM_PYTHON_PACKAGE}>")

file(GLOB EINSUM_PYTHON_SOURCES CONFIGURE_DEPENDS
        "einsum/*.py" "einsum/py.typed" "einsum/*.pyi")
add_custom_target(einsum_python_package ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${EINSUM_PYTHON_PACKAGE}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${EINSUM_PYTHON_SOURCES} "${EINSUM_PYTHON_PACKAGE}"
        DEPENDS ${EINSUM_PYTHON_SOURCES}
        COMMENT "Staging the einsum package beside _einsum")
add_dependencies(einsum_python_package einsum_python)

# --- pytest, as a ctest ------------------------------------------------------
set(EINSUM_PYTEST_ENV
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
        "PYTHONDONTWRITEBYTECODE=1"
        "EINSUM_PACKAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}")

# Under a sanitizer the interpreter is the process, and it is not the one that
# was instrumented -- so the runtime has to arrive first, by LD_PRELOAD.
# libstdc++ goes with it, and is not optional: ASan resolves the real
# __cxa_throw when it initialises, and python links no libstdc++ at startup, so
# without this the first throw out of the module is
#   CHECK failed: "((real___cxa_throw)) != (0)"
# and the process dies with no report at all.
if (NOT EINSUM_SANITIZE STREQUAL "off" AND NOT WIN32)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
            OUTPUT_VARIABLE EINSUM_LIBSTDCXX OUTPUT_STRIP_TRAILING_WHITESPACE)
    # address -> libasan, undefined -> libubsan, thread -> libtsan.
    set(EINSUM_SANITIZE_RUNTIMES address asan undefined ubsan thread tsan)
    list(FIND EINSUM_SANITIZE_RUNTIMES "${EINSUM_SANITIZE}" EINSUM_SANITIZE_AT)
    math(EXPR EINSUM_SANITIZE_AT "${EINSUM_SANITIZE_AT} + 1")
    list(GET EINSUM_SANITIZE_RUNTIMES ${EINSUM_SANITIZE_AT} EINSUM_SANITIZE_LIB)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER}
            -print-file-name=lib${EINSUM_SANITIZE_LIB}.so
            OUTPUT_VARIABLE EINSUM_LIBSAN OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND EINSUM_PYTEST_ENV
            "LD_PRELOAD=${EINSUM_LIBSAN}:${EINSUM_LIBSTDCXX}"
            "ASAN_OPTIONS=detect_leaks=0")
endif ()

add_test(NAME python.pytest
        COMMAND "${Python_EXECUTABLE}" -m pytest -q
        "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set_tests_properties(python.pytest PROPERTIES ENVIRONMENT "${EINSUM_PYTEST_ENV}")

# --- the comparison suite, on request ----------------------------------------
# Not a ctest: it takes minutes, it needs `uv sync --group bench`, and a number
# is neither a pass nor a fail.  The Python counterpart of benchmark/'s
# benchmark_json target.  SKBUILD is set by scikit-build-core, where the wheel
# build turns EINSUM_BUILD_PYTHON on as well and none of this is its business.
# The thread pools have to be pinned in the environment: OpenBLAS sizes its
# pool when it loads, so benchmarks/conftest.py can only check, not fix.
if (NOT SKBUILD)
    add_custom_target(python_benchmark_json
            COMMAND ${CMAKE_COMMAND} -E env ${EINSUM_PYTEST_ENV}
            "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1" "MKL_NUM_THREADS=1"
            # And the allocator, for the same reason as the thread pools: a
            # result big enough to be mmapped is handed back to the kernel on
            # free, so the next call faults in fresh zeroed pages.  That is a
            # real cost, but it is not one every process pays -- importing
            # torch alone retunes glibc and moves this suite's matmul_256 row
            # by 2.5x -- so it is pinned rather than left to whatever the
            # imports did.  glibc reads these at startup: they have to be in
            # the environment, which is why they are here and not in conftest.
            "MALLOC_MMAP_THRESHOLD_=67108864"
            "MALLOC_TRIM_THRESHOLD_=67108864"
            "${Python_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"
            --benchmark-group-by=param:case
            --benchmark-columns=min,median,ops
            --benchmark-sort=min
            --benchmark-warmup=on --benchmark-disable-gc
            --benchmark-json=${CMAKE_CURRENT_BINARY_DIR}/benchmark.json
            DEPENDS einsum_python_package
            COMMENT "Running the Python comparison suite into benchmark.json"
            VERBATIM)
endif ()

# --- stubs -------------------------------------------------------------------
# On request; the result is checked in.  Against the bare extension, not the
# package, so what it describes is what the C++ actually exports.
add_custom_target(einsum_python_stubs
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${EINSUM_PYTHON_PACKAGE}"
        "${Python_EXECUTABLE}" -m pybind11_stubgen _einsum
        -o "${CMAKE_CURRENT_BINARY_DIR}/stubs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/stubs/_einsum.pyi"
        "${CMAKE_CURRENT_SOURCE_DIR}/einsum/_einsum.pyi"
        DEPENDS einsum_python
        COMMENT "Regenerating python/einsum/_einsum.pyi")
