add_mlir_dialect(RockOps rock)
add_mlir_doc(RockOps RockDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(RockOps RockAttrDefs Dialects/ -gen-attrdef-doc)
add_mlir_doc(RockOps RockOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RockAttrDefs.td)
mlir_tablegen(RockTypes.h.inc -gen-enum-decls)
mlir_tablegen(RockTypes.cpp.inc -gen-enum-defs)
mlir_tablegen(RockAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(RockAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRRockAttrDefsIncGen)

add_mlir_interface(RockGemmWrapperInterface)
add_mlir_interface(RockConvInterface)
