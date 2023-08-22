add_mlir_dialect(MIGraphX migraphx)
add_mlir_doc(MIGraphX MIGraphXDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS MIGraphXTypes.td)
mlir_tablegen(MIGraphXEnums.h.inc -gen-enum-decls)
mlir_tablegen(MIGraphXEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMIGraphXTypeIncGen)
add_mlir_doc(MIGraphXTypes MIGraphXTypes ./ -gen-pass-doc)
