add_mlir_dialect(Dxgml dxgml)
add_mlir_doc(Dxgml DxgmlDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(DxgmlTypes DxgmlTypes Dialects/ -gen-typedef-doc -dialect=dxgml)

set(LLVM_TARGET_DEFINITIONS DxgmlAttrs.td)
mlir_tablegen(DxgmlAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(DxgmlAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRDxgmlAttrsIncGen)
add_dependencies(mlir-headers MLIRDxgmlAttrsIncGen)
