add_rocmlir_conversion_library(MLIRLinalgTosaToRockShared
  LinalgTosaToRockShared.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRRockOps
  MLIRRockUtility
  MLIRSupport
  MLIRTensorDialect
  MLIRTosaDialect
  )
