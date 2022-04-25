add_mlir_dialect_library(MLIRMIOpenUtility
  builderUtils.cpp
  loweringUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFunc
  MLIRMIOpenOps
  MLIRIR
  MLIRMemRef
  MLIRSupport
)

