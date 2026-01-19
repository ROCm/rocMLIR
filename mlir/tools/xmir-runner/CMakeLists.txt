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
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Desc
    LLVMX86Info
    MLIRMHAL
    MLIRMHALTransforms
    MLIRMHALPipeline
    MLIRParser
    MLIRRocmExecutionEngineUtils
    MLIRJitRunner
    MLIRGPUToLLVMIRTranslation
    MLIRAnalysis

    TritonIR
    TritonGPUIR
    TritonAnalysis
    TritonGPUTransforms
    TritonAMDGPUToLLVM
    TritonAMDGPUTransforms
    TritonAMDAnalysis
    TritonAMDGPUDialectToLLVM
    TritonAMDUtils
    GluonTransforms
    TritonInstrumentToLLVM
    NVHopperTransforms
    ProtonToProtonGPU
    ProtonGPUToLLVM
    ProtonNVIDIAGPUToLLVM
    ProtonAMDGPUToLLVM
    ProtonGPUTransforms
    TritonLLVMIR
    NVGPUToLLVM
    TritonInstrumentTransforms
  )

  add_rocmlir_tool(xmir-runner
    xmir-runner.cpp
    )

  target_link_libraries(xmir-runner PRIVATE ${LIBS})
  
  # Add Triton include directories for compilation
  target_include_directories(xmir-runner PRIVATE
    ${TRITON_PROJECT_DIR}/include
    ${TRITON_BINARY_DIR}/include
    ${TRITON_PROJECT_DIR}/third_party/amd
    ${TRITON_BINARY_DIR}/third_party/amd
    ${TRITON_PROJECT_DIR}/third_party/nvidia
    ${TRITON_BINARY_DIR}/third_party/nvidia
    ${TRITON_PROJECT_DIR}/third_party
    ${TRITON_PROJECT_DIR}
    ${TRITON_BINARY_DIR}/third_party
    ${TRITON_BINARY_DIR}/third_party/amd/include
    ${TRITON_BINARY_DIR}/third_party/nvidia/include
  )

  llvm_update_compile_flags(xmir-runner)

endif()
