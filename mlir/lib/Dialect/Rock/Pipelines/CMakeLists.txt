add_rocmlir_dialect_library(MLIRRockPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRGPUToROCDLTransforms
  MLIRTensorToLinalg
  MLIRTosaToArith
  MLIRTosaToRock
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToTensor
  MLIRRockToGPU
  MLIRRockOps
  MLIRRockTransforms
)
