
add_mlir_dialect_library(MLIRMHAL
  MHAL.cpp
  MHALOps.cpp

  DEPENDS
  MLIRMHALAttrDefsIncGen
  )
target_link_libraries(MLIRMHAL
  PUBLIC
  MLIRGPUOps
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRDialect
  MLIRAnalysis
  MLIRExecutionEngineUtils
  MLIRParser
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MLIRTransforms
  LLVMSupport
  )

