set(LLVM_OPTIONAL_SOURCES
  rocmlir-tuning-driver.cpp
  CacheFlush.cpp
  HipDelayLoad.cpp
  )

if(MLIR_ENABLE_ROCM_RUNNER OR ROCMLIR_BUILD_TUNING_DRIVER)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  MLIRParser
  MLIRMHALPipeline
)

add_rocmlir_tool(rocmlir-tuning-driver
  rocmlir-tuning-driver.cpp
  CacheFlush.cpp
  HipDelayLoad.cpp
)

# `rocmlir-tuning-driver` does NOT link `libamdhip64` / `libhiprtc` at
# build time. Doing so transitively pulls in `libamd_comgr` and ROCm's
# monolithic `libLLVM.so.<major>`, which fights with rocMLIR's own
# embedded LLVM at process start (`cl::SubCommand` static-init
# collision -> SmallPtrSet "Bucket < End" / "Option already exists").
# `HipDelayLoad.{h,cpp}` resolves every hipXXX/hiprtcXXX entry point at
# run time via dlmopen, coordinating with RocmSystemDetect through
# `mlirRocmSystemDetectGetHipHandle` so we share a single HSA session
# per process (KFD only allows one).
#
# We still need the HIP and HIPRTC headers (for type definitions like
# `hipDeviceProp_t`, `hipModule_t`, `hiprtcResult`, ...) and the
# `__HIP_PLATFORM_AMD__` define. Pull both from the standard `find_package`
# variables without consuming the imported targets' link components.
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
# load HIP, then remove it
set(REAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/hip" "${ROCM_PATH}/hiprtc")
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)
set(CMAKE_PREFIX_PATH "${REAL_CMAKE_PREFIX_PATH}")

target_include_directories(rocmlir-tuning-driver SYSTEM PRIVATE
  ${hip_INCLUDE_DIR})
target_compile_definitions(rocmlir-tuning-driver PRIVATE
  __HIP_PLATFORM_AMD__=1)
# `find_package(hiprtc)` exposes its include path either via the imported
# target's INTERFACE_INCLUDE_DIRECTORIES or via the legacy variable
# `hiprtc_INCLUDE_DIR`. Pick whichever is populated; both end up at the
# same `/opt/rocm/include`.
if (DEFINED hiprtc_INCLUDE_DIR)
  target_include_directories(rocmlir-tuning-driver SYSTEM PRIVATE
    ${hiprtc_INCLUDE_DIR})
endif()
if (TARGET hiprtc::hiprtc)
  get_target_property(_hiprtc_includes hiprtc::hiprtc
    INTERFACE_INCLUDE_DIRECTORIES)
  if (_hiprtc_includes)
    target_include_directories(rocmlir-tuning-driver SYSTEM PRIVATE
      ${_hiprtc_includes})
  endif()
  unset(_hiprtc_includes)
endif()

# Suppress HIP header warnings (HIP headers use GNU extensions that
# trigger -pedantic / -Wnested-anon-types / -Wreturn-type-c-linkage).
foreach(_src rocmlir-tuning-driver.cpp CacheFlush.cpp HipDelayLoad.cpp)
  set_source_files_properties(${_src} PROPERTIES
    COMPILE_OPTIONS
      "-Wno-c++98-compat-extra-semi;-Wno-return-type-c-linkage;-Wno-nested-anon-types;-Wno-gnu-anonymous-struct")
endforeach()

set_property(TARGET rocmlir-tuning-driver
  PROPERTY INSTALL_RPATH_USE_LINK_PATH ON)

# `MLIRRocmRuntimeLoader` is the shared static library that owns the
# dlmopen / dlopen / LoadLibraryW machinery (and pulls `${CMAKE_DL_LIBS}`
# transitively).
target_link_libraries(rocmlir-tuning-driver PRIVATE ${LIBS})
target_link_libraries(rocmlir-tuning-driver PRIVATE MLIRRocmRuntimeLoader)
llvm_update_compile_flags(rocmlir-tuning-driver)
endif()
