add_rocmlir_dialect_library(MLIRRockAnalysis
  BufferDependencyAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
)