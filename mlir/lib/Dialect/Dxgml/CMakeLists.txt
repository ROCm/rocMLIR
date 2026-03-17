add_subdirectory(DxgmlOp)

add_rocmlir_dialect_library(MLIRDxgmlDialect
  IR/Dxgml.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Dxgml
  ${PROJECT_BINARY_DIR}/mlir/include/mlir/Dialect/Dxgml/IR

  DEPENDS
  MLIRDxgmlIncGen
  MLIRDxgmlAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFunctionInterfaces
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
)
