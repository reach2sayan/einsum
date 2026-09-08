# Boost.Test's included variant, from the fetched Boost: no library to link and
# no libboost-dev on any machine that builds this.  Exceptions are on for these
# targets -- the whole of Boost.Test's reporting path is a throw.

# One suite per executable, since BOOST_TEST_MODULE is per program and
# tests_noalloc replaces operator new for the whole of its own.
set(EINSUM_TEST_SUITES tests_parse tests_ct tests_rt tests_noalloc)

# Not on MSVC.  Replacing operator new is program-wide on ELF, where the suite
# and libeinsum_rt resolve it to the one definition; a DLL keeps the CRT's.  So
# the count this suite asserts on would miss every allocation made behind
# einsum_rt.dll -- and worse, a block the DLL's new returned would reach the
# std::free in this suite's delete, which is the CRT's heap and not malloc's.
# The suite is a statement about the library's allocation behaviour, not about
# the platform, and Linux is where it can still be made.
if (MSVC)
    list(REMOVE_ITEM EINSUM_TEST_SUITES tests_noalloc)
endif ()

foreach (suite IN LISTS EINSUM_TEST_SUITES)
    add_executable(${suite} ${suite}.cpp)
    target_link_libraries(${suite} PRIVATE einsum::einsum einsum::rt)
    einsum_target_flags(${suite} EXCEPTIONS)
    einsum_target_pch(${suite})
    einsum_target_trace(${suite})
    einsum_runtime_deps(${suite})
    add_test(NAME ${suite} COMMAND ${suite})
endforeach ()

# Replacing operator new means GCC no longer sees which delete pairs with which
# new once Boost.Test's own allocations are inlined, and warns about every one.
# The pairing is correct; only the analysis cannot follow it.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(tests_noalloc PRIVATE -Wno-mismatched-new-delete)
endif ()

# --- the subscripts that must not compile ------------------------------------
# EXCLUDE_FROM_ALL, so a normal build never tries; the ctest below builds each
# one on purpose.
#
# PASS_REGULAR_EXPRESSION rather than WILL_FAIL: a test that only asks the build
# to fail passes when the build directory is broken, when an unrelated target
# does not compile, or when the compiler refuses for a reason nobody intended.
# Matching the sentence is what makes it a test of the diagnostic.
set(EINSUM_COMPILE_FAIL_CASES
        compile_fail_bad_subscript
        "the subscript has a character this grammar does not accept"
        compile_fail_extent_conflict
        "the output repeats a label"
        compile_fail_operand_count
        "the subscript names a different number of operands than the call passes"
        compile_fail_mixed_kinds
        "must be one family and one scalar"
        compile_fail_ellipsis_not_in_output
        "the operands' '...' covers axes the output does not name")

list(LENGTH EINSUM_COMPILE_FAIL_CASES _einsum_case_count)
math(EXPR _einsum_last "${_einsum_case_count} - 1")
foreach (i RANGE 0 ${_einsum_last} 2)
    list(GET EINSUM_COMPILE_FAIL_CASES ${i} case)
    math(EXPR j "${i} + 1")
    list(GET EINSUM_COMPILE_FAIL_CASES ${j} expected)

    add_executable(${case} EXCLUDE_FROM_ALL ${case}.cpp)
    target_link_libraries(${case} PRIVATE einsum::einsum)
    # No einsum_target_flags: -Werror would make an unrelated warning look like
    # the refusal this test is about.  /Zc:preprocessor is the exception, and
    # has to be named here for that same reason: the sentence these match is
    # built by Boost.Preprocessor, and without a conformant preprocessor the
    # build fails with a macro error instead of the static_assert -- a pass, by
    # a regex that never matched.
    target_compile_options(${case} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>)
    add_test(NAME ${case}
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${case} --config $<CONFIG>)
    set_tests_properties(${case} PROPERTIES PASS_REGULAR_EXPRESSION "${expected}")
endforeach ()
