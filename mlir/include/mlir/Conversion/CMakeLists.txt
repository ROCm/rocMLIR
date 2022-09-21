
set(LLVM_TARGET_DEFINITIONS RocMLIRPasses.td)
mlir_tablegen(RocMLIRPasses.h.inc -gen-pass-decls -name RocMLIRConversion)
add_public_tablegen_target(RocMLIRConversionPassIncGen)

add_mlir_doc(RocMLIRPasses -gen-pass-doc RocMLIRConversionPasses ./)
