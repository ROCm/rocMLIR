add_rocmlir_dialect_library(MLIRRockTransforms
  AffixTuningParameters.cpp
  BufferizableOpInterfaceImpl.cpp
  ConvToGemm.cpp
  DetectFlashDecoding.cpp
  FoldBroadcast.cpp
  GemmToGridwise.cpp
  Regularize.cpp
  GridwiseGemmToBlockwise.cpp
  GridLayoutEmitter.cpp
  ViewToTransform.cpp
  ShuffleGemmForReductions.cpp
  GemmLinalgSplitkNormalizationPass.cpp
  TransformsToPtr.cpp
  TransformsToPointerArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  DEPENDS
  MLIRRockOpsIncGen
  MLIRRockPassIncGen
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRLinalgUtils
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRMathToLLVM
  MLIRMemRefTransforms
  MLIRAffineToStandard
  MLIRReconcileUnrealizedCasts
  MLIRRockOps
  MLIRRockTuning
  MLIRRockUtility
  MLIRExecutionEngineUtils
  MLIRSCFTransforms
  MLIRAMDGPUUtils
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTosaDialect
  MLIRUBDialect
  MLIRVectorTransforms
  # MLIRMHAL  # TODO: Re-enable when MHAL updated for LLVM 22
  # MLIRMHALTransforms
)
