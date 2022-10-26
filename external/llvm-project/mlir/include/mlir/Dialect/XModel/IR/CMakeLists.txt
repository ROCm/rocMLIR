add_mlir_dialect(XModelBase xmodel)
add_mlir_doc(XModelBase XModelDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(XModelBase XModelAttrDefs Dialects/ -gen-attrdef-doc)
# add_mlir_doc(XModelOps XModelOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS XModelAttrDefs.td)
mlir_tablegen(XModelTypes.h.inc -gen-enum-decls)
mlir_tablegen(XModelTypes.cpp.inc -gen-enum-defs)
mlir_tablegen(XModelAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(XModelAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRXModelAttrDefsIncGen)

