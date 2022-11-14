add_rocmlir_dialect_library(MLIRRockPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRGPUToROCDLTransforms
  MLIRTensorToLinalg
  MLIRTosaToRock
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToArith
  MLIRRockToGPU
  MLIRRockOps
  MLIRRockTransforms
)
