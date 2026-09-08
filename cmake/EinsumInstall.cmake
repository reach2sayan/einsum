# Installing the library, and with it the third-party headers it needs.
#
# einsum is header-only but for one translation unit, so an installed copy is
# only usable if Boost, Eigen and mdspan travel with it: all three are fetched
# against a pin here, and none is a package a machine reliably has -- a distro's
# Boost predates Boost.Parser, and no libstdc++ on the floor ships <mdspan>.
# What ships is the part of each our own headers reach, which
# scripts/vendor_headers.py works out by asking the compiler.
#
# The result is one include directory holding einsum/, boost/, Eigen/,
# experimental/ and mdspan/, so a consumer needs no package but this one and the
# config file has no imported targets to reconstruct.
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(EinsumVendoredHeaders)

install(TARGETS einsum einsum_rt
        EXPORT einsumTargets
        FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# One dependency's subset, as whole directories one level deep.  A root named by
# an *_INCLUDEDIR override is the caller's own copy and stays theirs: they asked
# for it, they have it, and installing it under our prefix would shadow it.
function(_einsum_install_vendored label root dirs override)
    if (override)
        message(STATUS "einsum: not vendoring ${label}; ${root} is the caller's")
        return ()
    endif ()
    foreach (dir IN LISTS dirs)
        file(GLOB headers LIST_DIRECTORIES false "${root}/${dir}/*")
        if (NOT headers)
            message(FATAL_ERROR
                    "EinsumVendoredHeaders names ${dir}, which holds no files under "
                    "${root}.  The list and the tree disagree; regenerate it with "
                    "`python3 scripts/vendor_headers.py`.")
        endif ()
        install(FILES ${headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${dir}")
    endforeach ()
endfunction()

_einsum_install_vendored("Boost" "${EINSUM_BOOST_ROOT}"
        "${EINSUM_BOOST_HEADER_DIRS}" "${EINSUM_BOOST_INCLUDEDIR}")
_einsum_install_vendored("Eigen" "${EINSUM_EIGEN_ROOT}"
        "${EINSUM_EIGEN_HEADER_DIRS}" "${EINSUM_EIGEN_INCLUDEDIR}")
_einsum_install_vendored("mdspan" "${EINSUM_MDSPAN_ROOT}"
        "${EINSUM_MDSPAN_HEADER_DIRS}" "")

# --- the package -------------------------------------------------------------
install(EXPORT einsumTargets
        NAMESPACE einsum::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/einsum")

configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/einsum-config.cmake.in"
        "${PROJECT_BINARY_DIR}/einsum-config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/einsum")
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/einsum-config-version.cmake"
        COMPATIBILITY SameMajorVersion)
install(FILES
        "${PROJECT_BINARY_DIR}/einsum-config.cmake"
        "${PROJECT_BINARY_DIR}/einsum-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/einsum")

export(EXPORT einsumTargets NAMESPACE einsum::
        FILE "${PROJECT_BINARY_DIR}/einsumTargets.cmake")
