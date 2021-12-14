set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_tool(miopen-translate
  miopen-translate.cpp
  )
llvm_update_compile_flags(miopen-translate)
target_link_libraries(miopen-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${test_libs}
  MLIRIR
  MLIRMIGraphXToTosa
  MLIRGPUToMIGraphX
  MLIRMIOpenOps
  MLIRMIOpenThin
  MLIRMIOpenToGPU
  MLIRMIGraphX
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTosaToMIOpen
  MLIRTranslation
  MLIRSupport
  MLIRROCDLToLLVMIRTranslation
  )

mlir_check_link_libraries(miopen-translate)
