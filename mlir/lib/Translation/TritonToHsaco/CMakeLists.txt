llvm_canonicalize_cmake_booleans(
    MLIR_ENABLE_ROCM_CONVERSIONS
)

# Find LLD package for linking AMDGPU objects
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  find_package(LLD CONFIG QUIET)
  if(NOT LLD_FOUND)
    message(SEND_ERROR "LLD is not found, please ensure LLD was built with LLVM")
  endif()
endif()

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
  # X86 targets needed for InitializeAllTargets()
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info

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
  MLIRROCDLDialect
  MLIRRockOps

  # Triton libraries
  TritonAMDGPUToLLVM
  TritonLLVMIR  # For BreakStructPhiNodesPass
)

# Link LLD libraries for AMDGPU linking
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD AND LLD_FOUND)
  target_link_libraries(TritonToHsacoTranslation
    PRIVATE
    lldELF
    lldCommon
  )
endif()

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
