# --- the library -------------------------------------------------------------
# Shared, always.  It carries the two exported entry points and the Boost
# throw hook, and every consumer of it -- the Python extension included -- wants
# the versioned .so beside them rather than a copy in each.
add_library(einsum_rt SHARED boost_no_exceptions.cpp parse.cpp)
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
