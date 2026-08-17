# ---------------------------------------------------------------------------
# The Eternia paged-GEMM device library, built as an ExternalProject.
#
# Separate toolchain because the paging kernel suspends on a page fault using
# C++20 device coroutines, which only clang++-22 -x cuda compiles, while
# LBANN's CUDA is built by nvcc -- and nvcc rejects co_await in device code.
# The two halves meet at eternia_gemm.h, which names neither CUDA nor Clio.
# ---------------------------------------------------------------------------
include(ExternalProject)

set(LBANN_ETERNIA_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})
set(LBANN_ETERNIA_BIN_DIR ${CMAKE_BINARY_DIR}/eternia-build)
set(LBANN_ETERNIA_INS_DIR ${CMAKE_BINARY_DIR}/eternia-install)

set(LBANN_ETERNIA_CUDA_COMPILER "clang++-22" CACHE STRING
  "clang used for the Eternia device coroutines")

get_filename_component(_etn_clio_prefix "${iowarp-core_DIR}/../../.." ABSOLUTE)
set(_etn_prefix "${_etn_clio_prefix}")
foreach(_p IN LISTS CMAKE_PREFIX_PATH)
  string(APPEND _etn_prefix "|${_p}")
endforeach()

ExternalProject_Add(eternia_gemm_build
  SOURCE_DIR ${LBANN_ETERNIA_SRC_DIR}
  BINARY_DIR ${LBANN_ETERNIA_BIN_DIR}
  INSTALL_DIR ${LBANN_ETERNIA_INS_DIR}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${LBANN_ETERNIA_INS_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CUDA_COMPILER=${LBANN_ETERNIA_CUDA_COMPILER}
    -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
    -Diowarp-core_DIR=${iowarp-core_DIR}
    "-DCMAKE_PREFIX_PATH=${_etn_prefix}"
    -DETERNIA_LIB_ONLY=ON
  LIST_SEPARATOR |
  BUILD_BYPRODUCTS ${LBANN_ETERNIA_INS_DIR}/lib/liblbann_eternia.a
  # Without this the sub-build is stamped done once and never rerun, so
  # editing the kernel relinks against the previous archive and reports
  # success.
  BUILD_ALWAYS ON
)

add_library(LBANN::eternia STATIC IMPORTED)
set_target_properties(LBANN::eternia PROPERTIES
  IMPORTED_LOCATION ${LBANN_ETERNIA_INS_DIR}/lib/liblbann_eternia.a
  INTERFACE_INCLUDE_DIRECTORIES ${LBANN_ETERNIA_SRC_DIR})
add_dependencies(LBANN::eternia eternia_gemm_build)
