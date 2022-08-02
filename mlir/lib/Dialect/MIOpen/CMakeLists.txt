# Configure ROCm support.
if (NOT DEFINED ROCM_PATH)
  if (NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
  endif()
endif()

add_mlir_dialect_library(MLIRMIOpenOps
  TransformMapBuilder.cpp
  MIOpenDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/MIOpen

  DEPENDS
  MLIRMIOpenAttrDefsIncGen
  MLIRMIOpenOpsIncGen
  )
target_link_libraries(MLIRMIOpenOps
  PUBLIC
  MLIRGPUOps
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRDialect
  MLIRJitRunner
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRParser
  MLIRROCDLDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MLIRTransforms
  LLVMSupport
  )

add_subdirectory(Transforms)
add_subdirectory(Tuning)
add_subdirectory(Generator)
add_subdirectory(Pipelines)
add_subdirectory(utility)

