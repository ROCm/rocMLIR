add_rocmlir_conversion_library(MLIRMIGraphXToLinalg
  MIGraphXToLinalg.cpp
  MIGraphXToLinalgPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRMHAL
  MLIRTosaTransforms
  MLIRRockTransforms
  MLIRTransforms
  MLIRSupport
  MLIRMIGraphXDialect
  MLIRMIGraphXToTosa
  )
