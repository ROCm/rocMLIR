
add_mlir_dialect_library(MLIRXModel
  XModel.cpp

  DEPENDS
  MLIRXModelAttrDefsIncGen
  )
target_link_libraries(MLIRXModel
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

