add_rocmlir_dialect_library(MLIRRockUtility
  AccelEmitter.cpp
  builderUtils.cpp
  LdsTransposeLoad.cpp
  loweringUtils.cpp
  transformMapUtils.cpp
  fusionUtils.cpp
  tosaUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRArithDialect
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRRockOps
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRRockAnalysis
  # MLIRMHAL  # TODO: Re-enable when MHAL updated for LLVM 22
  MLIRLLVMDialect
)

add_subdirectory(Bindings)
