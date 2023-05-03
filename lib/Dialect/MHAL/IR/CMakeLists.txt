
add_mlir_dialect_library(MLIRMHAL
  MHAL.cpp
  MHALOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MHAL_MAIN_INCLUDE_DIR}/mlir/Dialect/MHAL

  DEPENDS
  MLIRMHALOpsIncGen
  MLIRMHALAttrDefsIncGen
  )

target_link_libraries(MLIRMHAL
  PUBLIC
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRCallInterfaces
  MLIRAnalysis
  MLIRParser
  MLIRSupport
  MLIRTransforms
  LLVMSupport
  )

