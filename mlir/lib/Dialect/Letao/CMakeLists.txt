file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRLetaoOps
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Letao

  DEPENDS
  MLIRLetaoOpsIncGen
  )

target_link_libraries(MLIRLetaoOps
  PUBLIC
  MLIRIR
  LLVMSupport
  )