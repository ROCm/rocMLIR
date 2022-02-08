add_mlir_dialect_library(MLIRMIOpenUtility
  builderUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRStandard
  MLIRSupport
)

