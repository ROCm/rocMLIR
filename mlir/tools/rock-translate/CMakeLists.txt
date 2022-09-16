set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_tool(rock-translate
  rock-translate.cpp
  )
llvm_update_compile_flags(rock-translate)
target_link_libraries(rock-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${test_libs}
  MLIRIR
  MLIRMIGraphXToTosa
  MLIRGPUToMIGraphX
  MLIRRockOps
  MLIRRockThin
  MLIRRockToGPU
  MLIRMIGraphX
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTosaToRock
  MLIRTranslateLib
  MLIRSupport
  MLIRROCDLToLLVMIRTranslation
  GpuModuleToRocdlirTranslation
  )

mlir_check_link_libraries(rock-translate)
