set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_tool(mlir-miopen-translate
  mlir-translate.cpp
  )
llvm_update_compile_flags(mlir-miopen-translate)
target_link_libraries(mlir-miopen-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${test_libs}
  MLIRIR
  MLIRMIOpenOps
  MLIRMIOpenThin
  MLIRMIOpenToGPU
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  MLIRTargetMIOpenCppTranslation
  )

mlir_check_link_libraries(mlir-miopen-translate)

# Use a new/unique target name, but use the typical executable name
set_target_properties(
  mlir-miopen-translate
  PROPERTIES
    OUTPUT_NAME "mlir-translate"
)
