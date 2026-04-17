add_rocmlir_conversion_library(MLIRLinalgToRock
  LinalgToRock.cpp
  LinalgToRockPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRRockTransforms
  MLIRTosaToRock
  MLIRTransforms
  MLIRSupport
  MLIRMIGraphXDialect
  )
