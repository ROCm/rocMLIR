add_rocmlir_conversion_library(MLIRTosaToRock
  TosaToRock.cpp
  TosaToRockPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRRockPassIncGen
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRRockConvGenerator
  MLIRRockTransforms
  MLIRLinalgUtils
  MLIRPass
  MLIRTosaToTensor
  MLIRTosaDialect
  MLIRTensorDialect
  MLIRTosaTransforms
  MLIRSupport
  )
