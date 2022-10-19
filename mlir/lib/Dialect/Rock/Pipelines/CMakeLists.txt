add_mlir_dialect_library(MLIRRockPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFuncDialect
  MLIRTensorToLinalg
  MLIRTosaToRock
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToArith
  MLIRLinalgToLLVM
  MLIRRockToGPU
  MLIRGPUToROCDLTransforms
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineToStandard
  MLIRRockOps
  MLIRRockTuning
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
