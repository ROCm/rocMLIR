add_mlir_dialect(MIOpenOps miopen)
add_mlir_doc(MIOpenOps MIOpenDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(MIOpenOps MIOpenAttrDefs Dialects/ -gen-attrdef-doc)
add_mlir_doc(MIOpenOps MIOpenOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MIOpenAttrDefs.td)
mlir_tablegen(MIOpenAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(MIOpenAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMIOpenAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRMIOpenPassIncGen)

add_mlir_doc(Passes MIOpenPasses ./ -gen-pass-doc)
