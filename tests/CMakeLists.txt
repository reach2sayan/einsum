# Boost.Test's included variant, from the fetched Boost: no library to link and
# no libboost-dev on any machine that builds this.  Exceptions are on for these
# targets -- the whole of Boost.Test's reporting path is a throw.

# One suite per executable, since BOOST_TEST_MODULE is per program and
# tests_noalloc replaces operator new for the whole of its own.
set(EINSUM_TEST_SUITES tests_parse tests_ct tests_rt tests_noalloc)

foreach (suite IN LISTS EINSUM_TEST_SUITES)
    add_executable(${suite} ${suite}.cpp)
    target_link_libraries(${suite} PRIVATE einsum::einsum einsum::rt)
    einsum_target_flags(${suite} EXCEPTIONS)
    einsum_target_pch(${suite})
    einsum_target_trace(${suite})
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
        "must be one family and one scalar")

list(LENGTH EINSUM_COMPILE_FAIL_CASES _einsum_case_count)
math(EXPR _einsum_last "${_einsum_case_count} - 1")
foreach (i RANGE 0 ${_einsum_last} 2)
    list(GET EINSUM_COMPILE_FAIL_CASES ${i} case)
    math(EXPR j "${i} + 1")
    list(GET EINSUM_COMPILE_FAIL_CASES ${j} expected)

    add_executable(${case} EXCLUDE_FROM_ALL ${case}.cpp)
    target_link_libraries(${case} PRIVATE einsum::einsum)
    # No einsum_target_flags: -Werror would make an unrelated warning look like
    # the refusal this test is about.
    add_test(NAME ${case}
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${case} --config $<CONFIG>)
    set_tests_properties(${case} PROPERTIES PASS_REGULAR_EXPRESSION "${expected}")
endforeach ()
