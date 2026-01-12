set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_rocmlir_tool(rocmlir-translate
  rocmlir-translate.cpp
  )
target_link_libraries(rocmlir-translate
  PRIVATE
  ${translation_libs}
  ${test_libs}
  # MLIRMHAL  # TODO: Re-enable when MHAL updated for LLVM 22
  # MLIRMHALTransforms
  # MLIRMHALPipeline
  MLIRROCDLToLLVMIRTranslation
  GpuModuleToRocdlirTranslation
  )

llvm_update_compile_flags(rocmlir-translate)
mlir_check_link_libraries(rocmlir-translate)
