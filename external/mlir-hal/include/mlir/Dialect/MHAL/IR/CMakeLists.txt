add_mlir_dialect(MHALOps mhal)
add_mlir_doc(MHALOps MHALDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS MHALAttrDefs.td)
mlir_tablegen(MHALTypes.h.inc -gen-enum-decls)
mlir_tablegen(MHALTypes.cpp.inc -gen-enum-defs)
mlir_tablegen(MHALAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(MHALAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMHALAttrDefsIncGen)

