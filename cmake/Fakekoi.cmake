if(NOT DORADO_ROCM_BUILD)
    return()
endif()

set(FAKEKOI_DIR "${DORADO_3RD_PARTY_SOURCE}/fakekoi")
if(NOT EXISTS "${FAKEKOI_DIR}/include/fakekoi.h")
    message(FATAL_ERROR "fakekoi not found at ${FAKEKOI_DIR}. Please initialise submodules.")
endif()

add_library(fakekoi STATIC
    ${FAKEKOI_DIR}/src/fakekoi.c
)
target_include_directories(fakekoi
    PUBLIC
        ${FAKEKOI_DIR}/include
)
