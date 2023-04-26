add_mlir_dialect_library(MLIRXModelPipeline
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
  MLIRXModelToGPU
  MLIRXModelTransforms
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
