add_mlir_dialect_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AlignTiling.cpp
  ApplyImpl.cpp
  AsyncLaunch.cpp
  BlockwiseGemmToThreadwise.cpp
  CleanMath.cpp
  CloneKernels.cpp
  CopyOpt.cpp
  ConvToGemm.cpp
  SugarToLoops.cpp
  GridwiseGemmToBlockwise.cpp
  LoopsToCf.cpp
  ThreadwiseGemmLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen

  DEPENDS
  MLIRMIOpenOpsIncGen
  MLIRMIOpenPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticTransforms
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRMIOpenUtility
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)

