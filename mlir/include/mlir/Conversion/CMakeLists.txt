
set(LLVM_TARGET_DEFINITIONS MIOpenPasses.td)
mlir_tablegen(MIOpenPasses.h.inc -gen-pass-decls -name MIOpenConversion)
add_public_tablegen_target(MLIRMIOpenConversionPassIncGen)

add_mlir_doc(MIOpenPasses -gen-pass-doc MIOpenConversionPasses ./)
