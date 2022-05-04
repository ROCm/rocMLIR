set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  LLVMAMDGPUAsmParser
  LLVMX86AsmParser
  LLVMROCmBackendUtils
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRMIOpenOps
  MLIRMIOpenPipeline
  MLIRParser
  MLIRPass
  MLIRQuant
  MLIRShape
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRTestDialect
  MLIRMIOpenThin
  )

add_llvm_executable(mlir-miopen-driver
  PARTIAL_SOURCES_INTENDED

  mlir-miopen-driver.cpp

  DEPENDS
  ${LIBS}
)


llvm_update_compile_flags(mlir-miopen-driver)
target_link_libraries(mlir-miopen-driver PRIVATE ${LIBS})
mlir_check_link_libraries(mlir-miopen-driver)

# Performance test scripts
# Defined here since mlir/utils/ isn't in Jenkins
file(GLOB PERFORMANCE_SCRIPTS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../utils/jenkins/*.py"
)

add_custom_target(ci-performance-scripts
	COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_SCRIPTS} ${MLIR_MIOPEN_BIN_DIR})
