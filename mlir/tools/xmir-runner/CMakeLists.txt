set(LLVM_OPTIONAL_SOURCES
  xmir-runner.cpp
  )

if(MLIR_ENABLE_ROCM_RUNNER)


  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    LLVMX86AsmParser
    MLIRJitRunner
    MLIRAnalysis
    MLIRExecutionEngine
    MLIRIR
    MLIRRockOps
    MLIRParser
    MLIRSupport
    MLIRGPUTransforms
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRTestDialect
    MLIRTransforms
    MLIRTranslateLib
    MLIRRocmExecutionEngineUtils
  )

  add_llvm_tool(xmir-runner
    xmir-runner.cpp
    )

  llvm_update_compile_flags(xmir-runner)
  target_link_libraries(xmir-runner PRIVATE ${LIBS})

endif()
