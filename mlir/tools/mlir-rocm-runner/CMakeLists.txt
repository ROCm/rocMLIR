set(LLVM_OPTIONAL_SOURCES
  mlir-rocm-runner.cpp
  )

if(MLIR_ENABLE_ROCM_RUNNER)

  # requirements for HIP
  if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building the mlir rocm runner requires the AMDGPU backend")
  endif()

  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/hip")
  find_package(hip REQUIRED)
  message("Found HIP version: ${hip_VERSION}")

  set_source_files_properties(mlir-rocm-runner.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-gnu-anonymous-struct;-Wno-nested-anon-types;-Wno-c++98-compat-extra-semi")

  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    LLVMX86AsmParser
    LLVMAMDGPUAsmParser
    MLIRJitRunner
    MLIRAnalysis
    MLIRExecutionEngine
    MLIRIR
    MLIRRockOps
    MLIRRockUtility
    MLIRParser
    MLIRROCDLDialect
    MLIRSupport
    MLIRGPUTransforms
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRTestDialect
    MLIRTransforms
    MLIRTranslateLib
  )

  add_llvm_tool(mlir-rocm-runner
    mlir-rocm-runner.cpp

    DEPENDS
    mlir_rocm_runtime
    conv-validation-wrappers
    )
  target_link_libraries(mlir-rocm-runner PRIVATE ${LIBS})
  target_link_libraries(mlir-rocm-runner PUBLIC hip::host hip::amdhip64)

endif()
