add_mlir_dialect_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AffineTransforms.cpp
  LowerMIOpenOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRMIOpenPassIncGen
  MLIRMIOpenTuning
  MLIRSCFToStandard
  MLIRStandardOps
)

target_link_libraries(MLIRMIOpenTransforms
  PUBLIC
  MLIRAffineToStandard
  MLIRIR
  MLIRLLVMIR
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRPass
  MLIRSCFToStandard
  MLIRStandardOps
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport
  )
