if(NOT DORADO_ROCM_BUILD)
    return()
endif()

set(OPENFISH_DIR "${DORADO_3RD_PARTY_SOURCE}/openfish")
if(NOT EXISTS "${OPENFISH_DIR}/include/openfish.h")
    message(FATAL_ERROR "openfish not found at ${OPENFISH_DIR}. Please initialise submodules.")
endif()

add_library(openfish STATIC
    ${OPENFISH_DIR}/src/openfish.c
)
target_include_directories(openfish
    PUBLIC
        ${OPENFISH_DIR}/include
)
