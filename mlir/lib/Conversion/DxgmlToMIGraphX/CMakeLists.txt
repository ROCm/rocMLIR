add_rocmlir_conversion_library(MLIRDxgmlToMIGraphX
  DxgmlToMIGraphX.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Dxgml
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIGraphX
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRFuncDialect
  MLIRDxgmlDialect
  MLIRDxgmlOpDialect
  MLIRMIGraphXDialect
  )
