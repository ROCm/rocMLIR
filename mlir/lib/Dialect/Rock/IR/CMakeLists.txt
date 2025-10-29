
add_rocmlir_dialect_library(MLIRRockOps
  AmdArchDb.cpp
  GetRockInfo.cpp
  TransformMapBuilder.cpp
  RockDialect.cpp
  RockGemmFeaturesInterface.cpp
  RockGemmWrapperInterface.cpp
  RockGemmGemmWrapperInterface.cpp
  RockConvInterface.cpp
  RockTuningParamAttrInterface.cpp
  RockAccelTuningParamAttrInterface.cpp
  RockAcceptingViewOpInterface.cpp
  RockWriterOpInterface.cpp
  MfmaInsnGroup.cpp
  WmmaInsnGroup.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rock

  DEPENDS
  MLIRRockAttrDefsIncGen
  MLIRRockGemmFeaturesInterfaceIncGen
  MLIRRockGemmWrapperInterfaceIncGen
  MLIRRockGemmGemmWrapperInterfaceIncGen
  MLIRRockTuningParamAttrInterfaceIncGen
  MLIRRockAccelTuningParamAttrInterfaceIncGen
  MLIRRockAcceptingViewOpInterfaceIncGen
  MLIRRockWriterOpInterfaceIncGen
  MLIRRockOpsIncGen
  )


target_link_libraries(MLIRRockOps
  PUBLIC
  MLIRAffineDialect
  MLIRAMDGPUDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRDialect
  MLIRAnalysis
  MLIRParser
  MLIRROCDLDialect
  MLIRSupport
  MLIRLinalgTransformOps
  )

# ROCm on Windows does not support HSA.
# Also HIP is not supported on our CI,
# so we cannot use either of them.
if (NOT WIN32)
  # From: https://github.com/ROCm/rocMLIR/blob/b6c726a324d2807494f770ed915af0e54cc6cb3f/external/llvm-project/mlir/lib/ExecutionEngine/CMakeLists.txt#L414
  find_package(hip REQUIRED PATHS ${ROCM_PATH})
  # We do need both of these. One is for linking, to get the HIP
  # functions.  The other is for compiling, to get -D__HIP_PLATFORM_AMD__
  target_link_libraries(MLIRRockOps PUBLIC
    hip::host hip::amdhip64
  )
  target_link_libraries(obj.MLIRRockOps PUBLIC
    hip::host hip::amdhip64
  )

  # From: https://github.com/ROCm/rocm-systems/blob/c53bdb9643afc2ef943fd25113a1c704d39aa6b4/projects/rocminfo/CMakeLists.txt
  find_package(hsa-runtime64 1.0 REQUIRED)
  target_link_libraries(MLIRRockOps PUBLIC hsa-runtime64::hsa-runtime64)
endif()
