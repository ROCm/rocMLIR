add_mlir_dialect_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AlignTiling.cpp
  BlockwiseGemmToThreadwise.cpp
  CopyOpt.cpp
  ConvToGemmLowering.cpp
  GridwiseGemmToBlockwise.cpp
  ApplyImpl.cpp
  CloneKernels.cpp
  ThreadwiseToStdlib.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen

  DEPENDS
  MLIRMIOpenOpsIncGen
  MLIRMIOpenPassIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRPass
  MLIRLLVMIR
  MLIRStandard
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRMIOpenUtility
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)

