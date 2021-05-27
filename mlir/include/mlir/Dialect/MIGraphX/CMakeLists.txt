add_mlir_dialect(MIGraphXOps migraphx)
#add_mlir_doc(MIOpenOps -gen-dialect-doc MIOpenDialect Dialects/)

#set(LLVM_TARGET_DEFINITIONS Passes.td)
#mlir_tablegen(Passes.h.inc -gen-pass-decls)
#add_public_tablegen_target(MLIRMIOpenPassIncGen)

#add_mlir_doc(Passes -gen-pass-doc MIOpenPasses ./)
