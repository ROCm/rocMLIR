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

set(LLVM_TARGET_DEFINITIONS RockTuningParamAttrInterface.td)
mlir_tablegen(RockTuningParamAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(RockTuningParamAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRRockTuningParamAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS RockAccelTuningParamAttrInterface.td)
mlir_tablegen(RockAccelTuningParamAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(RockAccelTuningParamAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRRockAccelTuningParamAttrInterfaceIncGen)

add_mlir_interface(RockGemmWrapperInterface)
add_mlir_interface(RockConvInterface)
