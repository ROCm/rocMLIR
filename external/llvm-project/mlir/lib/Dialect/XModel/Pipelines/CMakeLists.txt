add_mlir_dialect_library(MLIRXModelPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFuncDialect
  MLIRTensorToLinalg
  MLIRAsyncToGPU
  MLIRAsyncTransforms
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToArith
  MLIRLinalgToLLVM
  MLIRMathToLLVM
  MLIRGPUToROCDLTransforms
  MLIRReconcileUnrealizedCasts
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
