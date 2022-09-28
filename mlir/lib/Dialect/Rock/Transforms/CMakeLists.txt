add_mlir_dialect_library(MLIRRockTransforms
  AffixTuningParameters.cpp
  AlignTiling.cpp
  ApplyImpl.cpp
  AsyncLaunch.cpp
  BlockwiseGemmToThreadwise.cpp
  BufferizableOpInterfaceImpl.cpp
  CleanMath.cpp
  CloneKernels.cpp
  CopyOpt.cpp
  ConvToGemm.cpp
  FoldTranspose.cpp
  GemmToGridwise.cpp
  SugarToLoops.cpp
  GridwiseGemmToBlockwise.cpp
  LoopsToCf.cpp
  ThreadwiseGemmLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  DEPENDS
  MLIRRockOpsIncGen
  MLIRRockPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineToStandard
  MLIRRockOps
  MLIRRockTuning
  MLIRRockUtility
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)

