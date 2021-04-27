set(LLVM_OPTIONAL_SOURCES
  mlir-rocm-runner.cpp
  )

if(MLIR_ROCM_RUNNER_ENABLED)

  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    LLVMX86AsmParser
    LLVMAMDGPUAsmParser
    LLVMROCmBackendUtils
    MLIRJitRunner
    MLIRAnalysis
    MLIREDSC
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRROCDLIR
    MLIRSupport
    MLIRTargetLLVMIR
    MLIRTargetROCDLIR
    MLIRTransforms
    MLIRTranslation
  )

  add_llvm_tool(mlir-rocm-runner
    mlir-rocm-runner.cpp

    DEPENDS
    LLVMROCmBackendUtils
    rocm-runtime-wrappers
    ${LIBS}
    )
  llvm_update_compile_flags(mlir-rocm-runner)
  target_include_directories(mlir-rocm-runner
    PRIVATE
    "${HIP_PATH}/../include"
    "${HIP_PATH}/include"
  )
  target_link_libraries(mlir-rocm-runner PRIVATE ${LIBS})

endif()
