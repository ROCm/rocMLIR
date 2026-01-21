add_rocmlir_dialect_library(MLIRRockTransforms
  AffixTuningParameters.cpp
  BufferizableOpInterfaceImpl.cpp
  ConvToGemm.cpp
  DetectFlashDecoding.cpp
  FoldBroadcast.cpp
  GemmToGridwise.cpp
  Regularize.cpp
  GridwiseGemmToBlockwise.cpp
  GridLayoutEmitter.cpp
  ViewToTransform.cpp
  ShuffleGemmForReductions.cpp
  GemmLinalgSplitkNormalizationPass.cpp
  TransformsToPtr.cpp
  TransformsToPointerArith.cpp
  RockToTTIR.cpp
  MemrefToTensor.cpp
  Unbufferize.cpp
  RestoreHostCode.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  DEPENDS
  MLIRRockOpsIncGen
  MLIRRockPassIncGen
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRLinalgUtils
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRMathToLLVM
  MLIRMemRefTransforms
  MLIRAffineToStandard
  MLIRReconcileUnrealizedCasts
  MLIRRockOps
  MLIRRockTuning
  MLIRRockUtility
  MLIRExecutionEngineUtils
  MLIRSCFTransforms
  MLIRAMDGPUUtils
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTosaDialect
  MLIRUBDialect
  MLIRVectorTransforms

  # Triton dependencies
  # TritonIR depends on TritonGPUIR because Traits.cpp uses MemDescType
  # TritonGPUIR depends on TritonNvidiaGPUIR because it references NVIDIA types/ops
  # TritonAnalysis provides supportMMA, getReshapeDecomposition
  # TritonGPUTransforms provides inferDstEncoding, verifyBarrierType
  TritonGPUTransforms
  TritonIR
  TritonGPUIR
  TritonAnalysis
)

# Add Triton include directories for compilation
target_include_directories(MLIRRockTransforms PRIVATE
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_PROJECT_DIR}/third_party
  ${TRITON_BINARY_DIR}/third_party
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
