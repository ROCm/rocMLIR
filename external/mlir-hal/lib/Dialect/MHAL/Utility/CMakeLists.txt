add_mlir_dialect_library(MLIRMHALUtility
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRMHAL
  MLIRGPUDialect
)
