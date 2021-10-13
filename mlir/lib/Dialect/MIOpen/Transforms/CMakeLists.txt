add_mlir_dialect_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AffineTransforms.cpp
  AlignTiling.cpp
  LowerMIOpenOps.cpp
  CopyOpt.cpp
  TestAffineTransforms.cpp

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

