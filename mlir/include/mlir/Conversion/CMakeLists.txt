
set(LLVM_TARGET_DEFINITIONS RockPasses.td)
mlir_tablegen(RockPasses.h.inc -gen-pass-decls -name RockConversion)
add_public_tablegen_target(MLIRRockConversionPassIncGen)

add_mlir_doc(RockPasses -gen-pass-doc RockConversionPasses ./)
