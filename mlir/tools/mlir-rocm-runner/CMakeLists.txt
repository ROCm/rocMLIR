set(LLVM_OPTIONAL_SOURCES
  mlir-rocm-runner.cpp
  )

if(MLIR_ENABLE_ROCM_RUNNER)

  # requirements for HIP
  if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building the mlir rocm runner requires the AMDGPU backend")
  endif()

  # Set compile-time flags for ROCm path.
  add_definitions(-D__ROCM_PATH__="${ROCM_PATH}")

  set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH " Path to which HIP has been installed")
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP)
  if (NOT HIP_FOUND)
    message(SEND_ERROR "Build the mlir rocm runner requires a working ROCm and HIP install")
  else()
    message(STATUS "ROCm HIP version: ${HIP_VERSION}")
  endif()

  # Locate HIP runtime library.
  find_library(ROCM_RUNTIME_LIBRARY amdhip64
    PATHS "${HIP_ROOT_DIR}/lib")
  if (NOT ROCM_RUNTIME_LIBRARY)
    message(SEND_ERROR "Could not locate ROCm HIP runtime library")
  else()
    message(STATUS "ROCm HIP runtime lib: ${ROCM_RUNTIME_LIBRARY}")
  endif()

  # Set HIP compile-time flags.
  add_definitions(-D__HIP_PLATFORM_HCC__)
  add_definitions(-D__INCLUDE_HIP__=1)

  set_source_files_properties(mlir-rocm-runner.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-gnu-anonymous-struct;-Wno-nested-anon-types;-Wno-c++98-compat-extra-semi")

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
    MLIRExecutionEngine
    MLIRIR
    MLIRMIOpenOps
    MLIRParser
    MLIRROCDLIR
    MLIRSupport
    MLIRGPUTransforms
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRTestDialect
    MLIRTransforms
    MLIRTranslateLib
    ${ROCM_RUNTIME_LIBRARY}
  )

  add_llvm_tool(mlir-rocm-runner
    mlir-rocm-runner.cpp

    DEPENDS
    mlir_rocm_runtime
    rocm-runtime-wrappers
    )
  llvm_update_compile_flags(mlir-rocm-runner)
  target_include_directories(mlir-rocm-runner
    PRIVATE
    "${HIP_PATH}/../include"
    "${HIP_PATH}/include"
  )
  target_link_libraries(mlir-rocm-runner PRIVATE ${LIBS})

endif()
