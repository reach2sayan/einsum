# The grammar: the one object in the project compiled with exceptions, since
# Boost.Parser's parse() is a try/catch.  An OBJECT library so it can carry its
# own flags without lifting -fno-exceptions off anything else.
add_library(einsum_parse_objs OBJECT parse.cpp)

set_target_properties(einsum_parse_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

target_compile_definitions(einsum_parse_objs PRIVATE EINSUM_BUILDING BOOST_PARSER_DISABLE_TRACE)
target_link_libraries(einsum_parse_objs PRIVATE einsum::einsum Boost::headers)
einsum_target_flags(einsum_parse_objs EXCEPTIONS)
# No PCH: the shared one is built -fno-exceptions and would not be reusable here
# anyway, and a Boost.Parser TU is ten seconds whatever else is precompiled.

# --- the library -------------------------------------------------------------
if (EINSUM_BUILD_SHARED_RT)
    add_library(einsum_rt SHARED boost_no_exceptions.cpp)
else ()
    add_library(einsum_rt STATIC boost_no_exceptions.cpp)
endif ()
add_library(einsum::rt ALIAS einsum_rt)

set_target_properties(einsum_rt PROPERTIES
        OUTPUT_NAME einsum_rt
        EXPORT_NAME rt
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})

target_compile_definitions(einsum_rt PRIVATE EINSUM_BUILDING)
target_link_libraries(einsum_rt PUBLIC einsum::einsum)
einsum_target_flags(einsum_rt)

target_sources(einsum_rt PRIVATE $<TARGET_OBJECTS:einsum_parse_objs>)
