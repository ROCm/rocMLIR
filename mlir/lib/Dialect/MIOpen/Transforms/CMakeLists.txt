add_mlir_dialect_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AffineTransforms.cpp
  LowerMIOpenOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRMIOpenPassIncGen
  MLIRStandardOps
  MLIRLoopToStandard
)

target_link_libraries(MLIRMIOpenTransforms
  PUBLIC
  MLIRIR
  MLIRLoopOps
  MLIRLLVMIR
  MLIRMIOpenOps
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport
  )
