add_rocmlir_dialect_library(MLIRRockUtility
  builderUtils.cpp
  loweringUtils.cpp
  transformMapUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRArithmeticDialect
  MLIRFuncDialect
  MLIRRockOps
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
)

