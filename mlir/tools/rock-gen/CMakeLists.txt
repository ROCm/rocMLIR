set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRTestDialect
    )
endif()

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${test_libs}
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

add_llvm_executable(rock-gen
  PARTIAL_SOURCES_INTENDED

  rock-gen.cpp

  DEPENDS
  ${LIBS}
)


llvm_update_compile_flags(rock-gen)
target_link_libraries(rock-gen PRIVATE ${LIBS})
mlir_check_link_libraries(rock-gen)
