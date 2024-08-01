add_rocmlir_conversion_library(MLIRMIGraphXToTosa
  MIGraphXToTosa.cpp
  MIGraphXToTosaPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRTosaDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMHAL
  MLIRTosaTransforms
  MLIRRockTransforms
  MLIRTransforms
  MLIRSupport
  MLIRMIGraphXDialect
  )
