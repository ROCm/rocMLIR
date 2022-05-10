add_mlir_dialect_library(MLIRMIOpenPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFunc
  LLVMROCmBackendUtils
  MLIRTosaToMIOpen
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToArith
  MLIRMIOpenToGPU
  MLIRGPUToROCDLTransforms
  MLIRIR
  MLIRPass
  MLIRLLVMIR
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)
