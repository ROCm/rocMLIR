add_mlir_dialect_library(MLIRMHALPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFuncDialect
  MLIRTensorToLinalg
  MLIRAsyncTransforms
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToArith
  MLIRLinalgToLLVM
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRGPUToROCDLTransforms
  MLIRReconcileUnrealizedCasts
  MLIRMHALTransforms
  MLIRMHALToGPU
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
