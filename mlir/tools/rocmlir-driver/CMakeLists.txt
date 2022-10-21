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
  MLIRXModelPipeline
  MLIRParser
  MLIRPass
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRRockThin
  )

add_llvm_executable(rocmlir-driver
  PARTIAL_SOURCES_INTENDED

  rocmlir-driver.cpp

  DEPENDS
  ${LIBS}
)


llvm_update_compile_flags(rocmlir-driver)
target_link_libraries(rocmlir-driver PRIVATE ${LIBS})
mlir_check_link_libraries(rocmlir-driver)
