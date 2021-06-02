add_mlir_dialect(MIGraphXOps migraphx)
add_mlir_doc(MIGraphXOps -gen-dialect-doc MIGraphXDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRMIGraphXPassIncGen)

add_mlir_doc(Passes -gen-pass-doc MIGraphXPasses ./)
