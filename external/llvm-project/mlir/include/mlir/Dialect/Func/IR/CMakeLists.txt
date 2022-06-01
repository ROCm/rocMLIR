set(LLVM_TARGET_DEFINITIONS FuncOps.td)
mlir_tablegen(FuncOps.h.inc -gen-op-decls)
mlir_tablegen(FuncOps.cpp.inc -gen-op-defs)
mlir_tablegen(FuncOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FuncOpsDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(FuncOpsTypes.h.inc -gen-enum-decls)
mlir_tablegen(FuncOpsTypes.cpp.inc -gen-enum-defs)
mlir_tablegen(FuncOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(FuncOpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRFuncOpsIncGen)

add_mlir_doc(FuncOps FuncOps Dialects/ -gen-op-doc)

