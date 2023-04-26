add_mlir_dialect(MHALBase xmodel)
add_mlir_doc(MHALBase MHALDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(MHALBase MHALAttrDefs Dialects/ -gen-attrdef-doc)
add_mlir_doc(MHALOps MHALOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MHALAttrDefs.td)
mlir_tablegen(MHALTypes.h.inc -gen-enum-decls)
mlir_tablegen(MHALTypes.cpp.inc -gen-enum-defs)
mlir_tablegen(MHALAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(MHALAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMHALAttrDefsIncGen)

