add_mlir_dialect_library(MLIRMIOpenPipeline
  Pipelines.cpp
  XMIRPipelines.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFunc
  MLIRTensorToLinalg
  MLIRAsyncToGPU
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
  MLIRTransforms
  MLIRTransformUtils
)
