set(LLVM_OPTIONAL_SOURCES
  xmir-runner.cpp
  )

if(MLIR_ENABLE_ROCM_RUNNER)


  get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
  get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)
  get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

  set(LIBS
    ${rocmlir_dialect_libs}
    ${rocmlir_conversion_libs}
    ${rocmlir_test_libs}
    LLVMX86AsmParser
    MLIRMHAL
    MLIRMHALTransforms
    MLIRMHALPipeline
    MLIRParser
    MLIRRocmExecutionEngineUtils
    MLIRJitRunner
  )

  add_llvm_tool(xmir-runner
    xmir-runner.cpp
    )

  llvm_update_compile_flags(xmir-runner)
  target_link_libraries(xmir-runner PRIVATE ${LIBS})

endif()
