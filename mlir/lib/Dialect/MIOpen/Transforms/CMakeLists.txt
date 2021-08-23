add_mlir_dialect_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AffineTransforms.cpp
  AlignTiling.cpp
  LowerMIOpenOps.cpp
  TestAffineTransforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRMIOpenPassIncGen
  MLIRMIOpenTuning
  MLIRSCFToStandard
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
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport
  )
