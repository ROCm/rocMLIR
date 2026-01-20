llvm_canonicalize_cmake_booleans(
    MLIR_ENABLE_ROCM_CONVERSIONS
)

add_mlir_translation_library(TritonToHsacoTranslation
  TritonToHsaco.cpp

  DEPENDS
  MLIRRockPassIncGen

  LINK_COMPONENTS
  Core
  IRReader
  Linker
  Passes
  Target
  MC
  MCParser
  AsmParser
  AMDGPUCodeGen
  AMDGPUAsmParser
  AMDGPUDesc
  AMDGPUInfo
  TransformUtils
  Instrumentation

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRPass
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRIR
  MLIRRocTarget  # Contains LLD linking already set up

  # Triton libraries
  TritonAMDGPUToLLVM
  TritonLLVMIR  # For BreakStructPhiNodesPass
)

# Add Triton include directories
target_include_directories(obj.TritonToHsacoTranslation PRIVATE
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_PROJECT_DIR}/third_party/amd/include
  ${TRITON_BINARY_DIR}/third_party/amd/include
)

# Add LLD include directory for compilation (header access)
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  # Find LLD include directory from LLVM source tree
  list(GET LLVM_INCLUDE_DIRS 0 LLVM_FIRST_INCLUDE_DIR)
  get_filename_component(LLVM_SRC_DIR "${LLVM_FIRST_INCLUDE_DIR}" DIRECTORY)
  get_filename_component(LLVM_PROJECT_DIR "${LLVM_SRC_DIR}" DIRECTORY)
  set(LLD_INCLUDE_DIR_TRITON "${LLVM_PROJECT_DIR}/lld/include")

  if(EXISTS "${LLD_INCLUDE_DIR_TRITON}")
    message(STATUS "TritonToHsaco: Found LLD include dir at ${LLD_INCLUDE_DIR_TRITON}")
    target_include_directories(obj.TritonToHsacoTranslation PRIVATE ${LLD_INCLUDE_DIR_TRITON})
  else()
    message(WARNING "TritonToHsaco: LLD include dir not found at ${LLD_INCLUDE_DIR_TRITON}")
  endif()
endif()
