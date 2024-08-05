add_rocmlir_dialect_library(MLIRRockPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRAMDGPUTransforms
  MLIRGPUToROCDLTransforms
  MLIRArithToAMDGPU
  MLIRComplexToLLVM
  MLIRIndexToLLVM
  MLIRMathTransforms
  MLIRTensorToLinalg
  MLIRTosaToRock
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToTensor
  MLIRRockToGPU
  MLIRRockOps
  MLIRRockTransforms
  MLIRRockUtility
  MLIRUBToLLVM
  RocmlirEmulateFp8ExtTrunc
  RocmlirTosaToLinalg

  # Upstream dialect libraries we need to load that we don't directly depend on
  MLIRBufferizationTransformOps
  MLIRTensorInferTypeOpInterfaceImpl
)
