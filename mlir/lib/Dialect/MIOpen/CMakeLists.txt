file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRMIOpenOps
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/MIOpen

  DEPENDS
  MLIRMIOpenOpsIncGen
  )
target_link_libraries(MLIRMIOpenOps
  PUBLIC
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  )

add_subdirectory(Transforms)
add_subdirectory(Tuning)
