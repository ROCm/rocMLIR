
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

# MLIRRockOps deliberately has no link-time dependency on `libamdhip64` /
# `libhsa-runtime64`. `AmdArchDb.cpp` needs the HIP and HSA headers to know
# the POD layout of `hipDeviceProp_t` and a handful of HSA enums, but it
# resolves the actual functions at runtime via `dlopen`/`LoadLibraryW`.
# Avoiding the link-time dependency keeps ROCm's `libamd_comgr` and its
# embedded `libLLVM.so.23` out of every binary that links `MLIRRockOps`;
# otherwise the two LLVM copies race for the `cl::opt` global registry at
# static-init time and abort the process.
#
# The loader (`dlopen` / `LoadLibraryW`) lets the dynamic linker find the
# runtime libraries via the standard search path (`LD_LIBRARY_PATH` / RPATH
# on POSIX, the DLL search order on Windows); no compile-time install path
# is baked in.
find_package(hip QUIET)
if (NOT hip_FOUND)
  message(FATAL_ERROR
    "MLIRRockOps needs the HIP headers to describe `hipDeviceProp_t` for "
    "the `rock.arch = \"native\"` delay-loader in AmdArchDb.cpp. Install "
    "the ROCm `hip` CMake package (e.g. set `CMAKE_PREFIX_PATH=/opt/rocm`).")
endif()
target_include_directories(obj.MLIRRockOps SYSTEM PRIVATE ${hip_INCLUDE_DIRS})

if (NOT WIN32)
  find_package(hsa-runtime64 QUIET)
  if (hsa-runtime64_FOUND)
    get_target_property(_hsa_include_dirs hsa-runtime64::hsa-runtime64
      INTERFACE_INCLUDE_DIRECTORIES)
    if (_hsa_include_dirs)
      target_include_directories(obj.MLIRRockOps SYSTEM PRIVATE
        ${_hsa_include_dirs})
    endif()
    unset(_hsa_include_dirs)
  else()
    message(STATUS
      "hsa-runtime64 CMake package not found; AmdArchDb.cpp will look for "
      "HSA headers on the default include path.")
  endif()
endif()
