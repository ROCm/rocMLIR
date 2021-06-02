# Configure ROCm support.
#if (NOT DEFINED ROCM_PATH)
#  if (NOT DEFINED ENV{ROCM_PATH})
#    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
#  else()
#    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
#  endif()
#endif()

file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRMIGraphXOps
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIGraphX

  DEPENDS
  MLIRMIGraphXOpsIncGen
  MLIRMIGraphXPassIncGen
  )
target_link_libraries(MLIRMIGraphXOps
  PUBLIC
  MLIRGPU
  MLIRIR
  LLVMSupport
  )
