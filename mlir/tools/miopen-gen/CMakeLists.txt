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
  MLIRMIOpenOps
  MLIRMIOpenPipeline
  MLIRMIOpenUtility
  MLIRParser
  MLIRPass
  MLIRQuant
  MLIRShape
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRMIOpenThin
  )

add_llvm_executable(miopen-gen
  PARTIAL_SOURCES_INTENDED

  miopen-gen.cpp

  DEPENDS
  ${LIBS}
)


llvm_update_compile_flags(miopen-gen)
target_link_libraries(miopen-gen PRIVATE ${LIBS})
mlir_check_link_libraries(miopen-gen)
