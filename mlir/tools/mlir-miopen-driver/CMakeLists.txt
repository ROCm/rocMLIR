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
  MLIRParser
  MLIRPass
  MLIRQuant
  MLIRShape
  MLIRSDBM
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRTargetMIOpenCppTranslation
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

# MIOpenDriver.py
add_custom_target(MIOpenDriver.py ALL
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/jenkins/MIOpenDriver.py ${LLVM_BINARY_DIR})
