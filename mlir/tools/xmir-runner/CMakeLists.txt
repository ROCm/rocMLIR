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
    MLIRMIOpenOps
    MLIRParser
    MLIRSupport
    MLIRGPUTransforms
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRTestDialect
    MLIRTransforms
    MLIRTranslateLib
  )

  add_llvm_tool(xmir-runner
    xmir-runner.cpp

    DEPENDS
    conv-validation-wrappers
    mlir_rocm_runtime
    )
  llvm_update_compile_flags(xmir-runner)
  target_include_directories(xmir-runner
    PRIVATE
    "${HIP_PATH}/../include"
    "${HIP_PATH}/include"
  )
  target_link_libraries(xmir-runner PRIVATE ${LIBS})

endif()
