add_rocmlir_dialect_library(MLIRDxgmlOpDialect
  IR/DxgmlOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Dxgml/DxgmlOp
  ${PROJECT_BINARY_DIR}/mlir/include/mlir/Dialect/Dxgml/DxgmlOp/IR

  DEPENDS
  MLIRDxgmlOpIncGen
  MLIRDxgmlOpEnumsIncGen
  MLIRDxgmlDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDxgmlDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
