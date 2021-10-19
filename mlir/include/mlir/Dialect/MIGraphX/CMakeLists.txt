add_mlir_dialect(MIGraphXOps migraphx)
add_mlir_doc(MIGraphXOps MIGraphXDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS MIGraphXTypes.td)
mlir_tablegen(MIGraphXTypes.h.inc -gen-enum-decls)
mlir_tablegen(MIGraphXTypes.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMIGraphXTypeIncGen)
add_mlir_doc(MIGraphXTypes MIGraphXTypes ./ -gen-pass-doc)
