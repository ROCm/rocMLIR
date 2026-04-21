
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

# MLIRRockOps deliberately has no link-time dependency on the AMD HIP or HSA
# runtimes. All HIP/HSA usage lives in the `mlir_rocm_arch_runtime` shared
# library (see mlir/lib/ExecutionEngine/RocmArchRuntime.cpp), which
# `AmdArchDb.cpp` loads on demand via a private `dlopen`/`LoadLibraryW` call.
# Keeping ROCm's `libLLVM.so`/`libamd_comgr` out of the main process avoids
# the `cl::opt` double-registration that otherwise corrupts `rocmlir-driver`
# and similar tools.
#
# Record the build-tree location of the runtime library so nested test
# binaries (e.g. unit tests deep under `build/mlir/unittests/...`) can find
# it without `LD_LIBRARY_PATH`. Install-tree consumers fall back to the
# `<exe-dir>/../lib/` search rule in `AmdArchDb.cpp`. On Windows the runtime
# artifact we need to load is the DLL, so use the runtime output directory;
# on POSIX the shared library lives in the library output directory.
#
# This must NOT be guarded by `if (TARGET mlir_rocm_arch_runtime)` because
# `mlir/lib/ExecutionEngine/CMakeLists.txt` (where that target is created)
# is processed AFTER this directory (see `add_subdirectory` order in
# `mlir/lib/CMakeLists.txt`). The output directory variables themselves are
# set globally by LLVM, so they are always available here.
set(_rocmlir_build_runtime_dir "${LLVM_LIBRARY_OUTPUT_INTDIR}")
if (WIN32)
  set(_rocmlir_build_runtime_dir "${LLVM_RUNTIME_OUTPUT_INTDIR}")
endif()
target_compile_definitions(MLIRRockOps PRIVATE
  "ROCMLIR_BUILD_RUNTIME_DIR=\"${_rocmlir_build_runtime_dir}\"")
target_compile_definitions(obj.MLIRRockOps PRIVATE
  "ROCMLIR_BUILD_RUNTIME_DIR=\"${_rocmlir_build_runtime_dir}\"")
unset(_rocmlir_build_runtime_dir)

# Build-only dependency: ensures the runtime shared library is produced before
# any consumer of MLIRRockOps. The dependency is wired up at the end of the
# top-level CMake (see `mlir/lib/CMakeLists.txt`) once the ExecutionEngine
# subdirectory has had a chance to declare the runtime target.
