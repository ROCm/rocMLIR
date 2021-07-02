set(LLVM_TARGET_DEFINITIONS <conversion.td>)
mlir_tablegen(<conversion.inc> -gen-rewriters)
add_public_tablegen_target(<MLIRTosaToMIGraphXConversionIncGen>)