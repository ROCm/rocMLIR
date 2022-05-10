add_mlir_dialect_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AlignTiling.cpp
  ApplyImpl.cpp
  AsyncLaunch.cpp
  BlockwiseGemmToThreadwise.cpp
  CloneKernels.cpp
  CopyOpt.cpp
  ConvToGemmLowering.cpp
  ExpandShorthand.cpp
  GridwiseGemmToBlockwise.cpp
  LoopsToCf.cpp
  ThreadwiseToStdlib.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen

  DEPENDS
  MLIRMIOpenOpsIncGen
  MLIRMIOpenPassIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFunc
  MLIRIR
  MLIRPass
  MLIRLLVMIR
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRMIOpenUtility
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)

