
set(LLVM_TARGET_DEFINITIONS MHALPasses.td)
mlir_tablegen(MHALPasses.h.inc -gen-pass-decls -name MHALConversion)
#mlir_tablegen(MHALPasses.capi.h.inc -gen-pass-capi-header --prefix MHALConversion)
#mlir_tablegen(MHALPasses.capi.cpp.inc -gen-pass-capi-impl --prefix MHALConversion)
add_public_tablegen_target(MHALConversionPassIncGen)

add_mlir_doc(MHALPasses MHALConversionPasses ./ -gen-pass-doc)
