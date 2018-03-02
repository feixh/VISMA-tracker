# Try to find the MPFR C++ bindings
# http://www.holoborodko.com/pavel/mpfr/
find_path(MPFR_CXX_INCLUDES
        NAMES
        mpreal.h
        PATHS
        $ENV{MPFR_CXX_DIR}
        ${INCLUDE_INSTALL_DIR}
        )
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR_CXX DEFAULT_MSG
        MPFR_CXX_INCLUDES )
mark_as_advanced(MPFR_CXX_INCLUDES )