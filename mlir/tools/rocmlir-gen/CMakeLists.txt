set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_test_libs}
  LLVMAMDGPUAsmParser
  LLVMX86AsmParser
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRRockOps
  MLIRRockPipeline
  MLIRRockUtility
  MLIRParser
  MLIRPass
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRRockThin
  )

add_llvm_executable(rocmlir-gen
  PARTIAL_SOURCES_INTENDED

  rocmlir-gen.cpp

  DEPENDS
  ${LIBS}
)


llvm_update_compile_flags(rocmlir-gen)
target_link_libraries(rocmlir-gen PRIVATE ${LIBS})
mlir_check_link_libraries(rocmlir-gen)
