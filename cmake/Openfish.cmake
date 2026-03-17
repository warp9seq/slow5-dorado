if(NOT DORADO_ROCM_BUILD)
    return()
endif()

set(OPENFISH_DIR "${DORADO_3RD_PARTY_SOURCE}/openfish")
if(NOT EXISTS "${OPENFISH_DIR}/include/openfish/openfish.h")
    message(FATAL_ERROR "openfish not found at ${OPENFISH_DIR}. Please initialise submodules.")
endif()

set(OPENFISH_USE_ROCM ON)

add_subdirectory(${OPENFISH_DIR} ${CMAKE_CURRENT_BINARY_DIR}/openfish)
