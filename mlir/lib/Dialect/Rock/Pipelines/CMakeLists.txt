add_rocmlir_dialect_library(MLIRRockPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRAMDGPUTransforms
  MLIRGPUToROCDLTransforms
  MLIRArithToAMDGPU
  MLIRComplexToLLVM
  MLIRIndexToLLVM
  MLIRMathTransforms
  MLIRTensorToLinalg
  MLIRTosaToRock
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToTensor
  MLIRRockOps
  MLIRRockTransforms
  MLIRRockUtility
  MLIRUBToLLVM
  RocmlirCustomTosaDecompose
  RocmlirCustomTosaToLinalg
  RocmlirEmulateFp8ExtTrunc

  # Upstream dialect libraries we need to load that we don't directly depend on
  MLIRBufferizationTransformOps
  MLIRTensorInferTypeOpInterfaceImpl
  
  TritonGPUTransforms
  TritonIR
  TritonGPUIR
  TritonAnalysis
)

# Add Triton include directories for compilation
target_include_directories(MLIRRockPipeline PRIVATE
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_PROJECT_DIR}/third_party/amd
  ${TRITON_BINARY_DIR}/third_party/amd
  ${TRITON_PROJECT_DIR}/third_party/nvidia
  ${TRITON_BINARY_DIR}/third_party/nvidia
)

# Fix build order: TritonIR and TritonGPUIR include headers that need generated files
# This ensures tablegen runs before these libraries build
if(TARGET TritonIR)
  foreach(target IN ITEMS
      TritonGPUTableGen
      TritonGPUAttrDefsIncGen
      TritonGPUCGAAttrIncGen
      TritonGPUTypeInterfacesIncGen
      TritonGPUOpInterfacesIncGen)
    if(TARGET ${target})
      add_dependencies(TritonIR ${target})
    endif()
  endforeach()
endif()

# TritonGPUIR unconditionally includes NVIDIA headers, so we need NVIDIA tablegen
if(TARGET TritonGPUIR)
  foreach(target IN ITEMS
      TritonNvidiaGPUTableGen
      TritonNvidiaGPUAttrDefsIncGen
      TritonNvidiaGPUOpInterfacesIncGen)
    if(TARGET ${target})
      add_dependencies(TritonGPUIR ${target})
    endif()
  endforeach()
endif()

# TritonGPUTransforms includes NVWS headers that need generated files
# This ensures NVWS tablegen runs before TritonGPUTransforms builds
if(TARGET TritonGPUTransforms)
  foreach(target IN ITEMS
      NVWSTableGen
      NVWSAttrDefsIncGen)
    if(TARGET ${target})
      add_dependencies(TritonGPUTransforms ${target})
    endif()
  endforeach()
endif()
