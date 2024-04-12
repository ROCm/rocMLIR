if (MLIR_ENABLE_ROCM_CONVERSIONS)
  set(AMDGPU_LIBS
    IRReader
    IPO
    linker
    MCParser
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
    target
  )
endif()

add_mlir_dialect_library(MLIRGPUDialect
  IR/GPUDialect.cpp
  IR/InferIntRangeInterfaceImpls.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  DEPENDS
  MLIRGPUOpsIncGen
  MLIRGPUOpsAttributesIncGen
  MLIRGPUOpsEnumsGen
  MLIRGPUOpInterfacesIncGen
  MLIRGPUCompilationAttrInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRInferIntRangeInterface
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  )

add_mlir_dialect_library(MLIRGPUTransforms
  Transforms/AllReduceLowering.cpp
  Transforms/AsyncRegionRewriter.cpp
  Transforms/BufferDeallocationOpInterfaceImpl.cpp
  Transforms/DecomposeMemrefs.cpp
  Transforms/EliminateBarriers.cpp
  Transforms/GlobalIdRewriter.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ModuleToBinary.cpp
  Transforms/NVVMAttachTarget.cpp
  Transforms/ParallelLoopMapper.cpp
  Transforms/ROCDLAttachTarget.cpp
  Transforms/SerializeToBlob.cpp
  Transforms/SerializeToHsaco.cpp
  Transforms/ShuffleRewriter.cpp
  Transforms/SPIRVAttachTarget.cpp
  Transforms/SubgroupReduceLowering.cpp
  Transforms/Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_COMPONENTS
  Core
  MC
  Target
  ${AMDGPU_LIBS}

  DEPENDS
  MLIRGPUPassIncGen
  MLIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRDataLayoutInterfaces
  MLIRExecutionEngineUtils
  MLIRGPUDialect
  MLIRIR
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRNVVMTarget
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSPIRVTarget
  MLIRSupport
  MLIRROCDLTarget
  MLIRTransformUtils
  MLIRVectorDialect
  )

add_subdirectory(TransformOps)
add_subdirectory(Pipelines)

if(MLIR_ENABLE_ROCM_CONVERSIONS)
  if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building mlir with ROCm support requires the AMDGPU backend")
  endif()

  if (NOT ("lld" IN_LIST LLVM_ENABLE_PROJECTS))
    message(SEND_ERROR
      "LLD is not enabled, please reconfigure llvm build")
  endif()

  ### rocMLIR-specific incantations to embed device libraries statically.
  ### This is menat to let us embed the device libraries into a static library.
  if (NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
    else()
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
  endif()
  # A lot of the ROCm CMake files expect to find their own dependencies in
  # CMAKE_PREFIX_PATH and don't respect PATHS or HINTS :( .
  # Therefore, temporarily add the ROCm path to CMAKE_PREFIX_PATH so we can
  # find the device libraries, then remove it
  set(REAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/hip")
  find_package(AMDDeviceLibs CONFIG)
  set(CMAKE_PREFIX_PATH "${REAL_CMAKE_PREFIX_PATH}")
  if(AMDDeviceLibs_FOUND)
    set(device_lib_targets ${AMD_DEVICE_LIBS_TARGETS})
    # Filter out control constants, since we set those inline
    list(FILTER device_lib_targets EXCLUDE REGEX "^oclc_")
    # None of our stuff uses opencl functions or functionality during
    # compilation, and the only time that device library gets pulled in is for
    # printf(), which we use the hip version of anyway.
    list(REMOVE_ITEM device_lib_targets "opencl")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Transforms)
    add_custom_command(OUTPUT Transforms/AmdDeviceLibs.cpp.inc
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/AmdDeviceLibsIncGen.py
      ARGS
        ${CMAKE_CURRENT_BINARY_DIR}/Transforms/AmdDeviceLibs.cpp.inc
        ${AMD_DEVICE_LIBS_PREFIX}
        ${device_lib_targets}
      DEPENDS
        AmdDeviceLibsIncGen.py
        ${AMD_DEVICE_LIBS_TARGETS}
      COMMENT "Generating device libraries include package"
    )
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES Transforms/AmdDeviceLibs.cpp.inc)
    add_custom_target(AmdDeviceLibsIncGen DEPENDS Transforms/AmdDeviceLibs.cpp.inc)
    set_property(SOURCE Transforms/SerializeToHsaco.cpp APPEND
      PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Transforms/AmdDeviceLibs.cpp.inc)
    add_dependencies(obj.MLIRGPUTransforms AmdDeviceLibsIncGen)
    target_compile_definitions(obj.MLIRGPUTransforms
      PRIVATE
      ROCMLIR_DEVICE_LIBS_PACKAGED=1
    )

  else()
    message(NOTICE "Device libraries not found during builds, assuming we can find them at runtime")
  endif()
  ### End rocMLIR-specific hackery
  set(DEFAULT_ROCM_PATH "/opt/rocm" CACHE PATH "Fallback path to search for ROCm installs")
  target_compile_definitions(obj.MLIRGPUTransforms
    PRIVATE
    __DEFAULT_ROCM_PATH__="${DEFAULT_ROCM_PATH}"
  )

  target_include_directories(obj.MLIRGPUTransforms
    PRIVATE
    ${MLIR_SOURCE_DIR}/../lld/include
  )

  target_link_libraries(MLIRGPUTransforms
    PRIVATE
    lldELF
    lldCommon
    MLIRROCDLToLLVMIRTranslation
  )

  # Link lldELF also to libmlir.so. Create an alias that starts with LLVM
  # because LINK_COMPONENTS elements are implicitly prefixed with LLVM.
  add_library(LLVMAliasTolldELF ALIAS lldELF)
  set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS AliasTolldELF)
endif()
