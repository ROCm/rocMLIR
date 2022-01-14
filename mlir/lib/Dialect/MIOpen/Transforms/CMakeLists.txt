add_mlir_dialect_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AlignTiling.cpp
  CopyOpt.cpp
  ConvToGemmLowering.cpp
  LowerMIOpenOps.cpp

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
  MLIRSCFToStandard
  MLIRSupport
  MLIRTransformUtils
)

