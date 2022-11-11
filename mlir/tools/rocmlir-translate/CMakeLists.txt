set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(rocm_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_tool(rocmlir-translate
  rocmlir-translate.cpp
  )
llvm_update_compile_flags(rocmlir-translate)
target_link_libraries(rocmlir-translate
  PRIVATE
  ${rocm_dialect_libs}
  ${translation_libs}
  ${test_libs}
  MLIRROCDLToLLVMIRTranslation
  GpuModuleToRocdlirTranslation
  )

mlir_check_link_libraries(rocmlir-translate)
