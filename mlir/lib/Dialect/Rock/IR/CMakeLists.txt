
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

# rock.arch="native" runtime detection in AmdArchDb.cpp is the only thing
# in MLIRRockOps that needs HIP and HSA at link time. Gating that behind
# ROCMLIR_ENABLE_NATIVE_ARCH lets BUILD_FAT_LIBROCKCOMPILER consumers (e.g.
# MIGraphX, which always passes a real arch like "gfx942") avoid a hard
# DT_NEEDED on libamdhip64/libhsa-runtime64 from rocMLIR. Windows is also
# excluded because HIP/HSA are not supported there.
if (NOT WIN32 AND ROCMLIR_ENABLE_NATIVE_ARCH)
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

  # Tell AmdArchDb.cpp to compile in the HIP/HSA-backed nativeArchInfo path.
  target_compile_definitions(obj.MLIRRockOps PRIVATE ROCMLIR_ENABLE_NATIVE_ARCH=1)
endif()
