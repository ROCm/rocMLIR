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
  MLIRParser
  MLIRPass
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRRockThin
  )

add_llvm_executable(mlir-rock-driver
  PARTIAL_SOURCES_INTENDED

  mlir-rock-driver.cpp

  DEPENDS
  ${LIBS}
)


llvm_update_compile_flags(mlir-rock-driver)
target_link_libraries(mlir-rock-driver PRIVATE ${LIBS})
mlir_check_link_libraries(mlir-rock-driver)

# Performance test scripts
# Defined here since mlir/utils/ isn't in Jenkins
file(GLOB PERFORMANCE_SCRIPTS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../utils/jenkins/*.py"
)

add_custom_target(ci-performance-scripts
	COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_SCRIPTS} ${ROCMLIR_BIN_DIR})
