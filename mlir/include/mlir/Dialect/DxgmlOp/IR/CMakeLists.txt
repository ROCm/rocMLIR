add_mlir_dialect(DxgmlOp dxgml_op)
add_mlir_doc(DxgmlOp DxgmlOpDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DxgmlOp.td)
mlir_tablegen(DxgmlOpEnums.h.inc -gen-enum-decls)
mlir_tablegen(DxgmlOpEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRDxgmlOpEnumsIncGen)
