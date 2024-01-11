add_rocmlir_dialect_library(MLIRRockPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRAMDGPUTransforms
  MLIRGPUToROCDLTransforms
  MLIRArithToAMDGPU
  MLIRTensorToLinalg
  MLIRTosaToArith
  MLIRTosaToRock
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRLinalgToLLVM
  MLIRTosaToSCF
  MLIRTosaToTensor
  MLIRRockToGPU
  MLIRRockOps
  MLIRRockTransforms
  MLIRRockUtility
  RocmlirEmulateFp8ExtTrunc

  # Upstream dialect libraries we need to load that we don't directly depend on
  MLIRBufferizationTransformOps
  MLIRTensorInferTypeOpInterfaceImpl
)
