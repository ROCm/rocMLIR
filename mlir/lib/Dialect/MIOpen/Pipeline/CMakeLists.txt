add_mlir_dialect_library(MLIRMIOpenPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  LLVMROCmBackendUtils
  MLIRTosaToMIOpen
  MLIRTosaToLinalg
  MLIRMIOpenToGPU
  MLIRGPUToROCDLTransforms
  MLIRIR
  MLIRPass
  MLIRLLVMIR
  MLIRStandard
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)
