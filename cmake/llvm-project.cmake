message(STATUS "Adding LLVM git-submodule src dependency")

set(LLVM_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project")
set(LLVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project")

# Pointers to: external LLVM bins/libs
set(LLVM_EXTERNAL_BIN_DIR "${LLVM_BINARY_DIR}/llvm/bin" CACHE PATH "")
set(LLVM_EXTERNAL_LIB_DIR "${LLVM_BINARY_DIR}/llvm/lib" CACHE PATH "")

message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "LLVM_EXTERNAL_LIB_DIR: ${LLVM_EXTERNAL_LIB_DIR}")

# Passed to lit.site.cfg.py.so that the out of tree Standalone dialect test
# can find MLIR's CMake configuration
set(MLIR_CMAKE_CONFIG_DIR
   "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")

# MLIR settings
set(MLIR_TABLEGEN_EXE mlir-tblgen)

# LLVM settings that have an effect on the MLIR dialect
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_PROJECTS "mlir;lld" CACHE STRING "List of default llvm targets")
set(LLVM_BUILD_EXAMPLES ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING" CACHE INTERNAL "")
endif()

# Only build the X86 backend if we'll be JIT-ing host code.
if (MLIR_ENABLE_ROCM_RUNNER)
  set(LLVM_TARGETS_TO_BUILD "X86;AMDGPU" CACHE STRING "")
else()
  set(LLVM_TARGETS_TO_BUILD "AMDGPU" CACHE STRING "")
endif()

# Configure ROCm support.
if (NOT DEFINED ROCM_PATH)
  if (NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
  endif()
endif()
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

# Cmake module paths
list(APPEND CMAKE_MODULE_PATH
  "${ROCM_PATH}/hip/cmake"
)
list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_BINARY_DIR}/lib/cmake/mlir"
)
list(APPEND CMAKE_MODULE_PATH
  "${LLVM_BINARY_DIR}/llvm/lib/cmake/llvm/"
)

# Include dirs for MLIR and LLVM
list(APPEND MLIR_INCLUDE_DIRS
  ${LLVM_PROJECT_DIR}/mlir/include
  ${LLVM_BINARY_DIR}/llvm/tools/mlir/include
)
list(APPEND LLVM_INCLUDE_DIRS
  ${LLVM_PROJECT_DIR}/llvm/include
  ${LLVM_BINARY_DIR}/llvm/include
)

# Linker flags
list(PREPEND CMAKE_BUILD_RPATH "${LLVM_EXTERNAL_LIB_DIR}")
### Workaround ROCm address sanitizer build not being able to propagate LD_LIBRARY_PATH
### Remove when https://github.com/pfultz2/cget/issues/110 is fixed.
if (ENV{ADDRESS_SANITIZER})
  execute_process(
    COMMAND ${CMAKE_C_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
    OUTPUT_VARIABLE clang_asan_lib_file
    ERROR_VARIABLE clang_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE clang_exit_code)
  if (NOT "${clang_exit_code}" STREQUAL "0")
    message(FATAL_ERROR
      "Unable to invoke clang to find asan lib dir: ${clang_stderr}")
  endif()
  file(TO_CMAKE_PATH "${clang_asan_lib_file}" clang_asan_lib_file)
  get_filename_component(clang_asan_lib_dir "${clag_asan_lib_file}" DIRECTORY)
  list(APPEND CMAKE_BUILD_RPATH "${clang_asan_lib_dir}")
endif()
### End workaround

add_subdirectory("${LLVM_PROJECT_DIR}/llvm" "external/llvm-project/llvm" EXCLUDE_FROM_ALL)

# Propagate the RPATH settings up to rocMLIR, since we need them there too
set(CMAKE_BUILD_RPATH ${CMAKE_BUILD_RPATH} PARENT_SCOPE)

function(add_rocmlir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_DIALECT_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_dialect_library)

function(add_rocmlir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_CONVERSION_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_conversion_library)

function(add_rocmlir_test_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_TEST_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_test_library)

function(add_rocmlir_public_c_api_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_PUBLIC_C_API_LIBS ${name})
  add_mlir_library(${name}
    ${ARGN}
    EXCLUDE_FROM_LIBMLIR
    ENABLE_AGGREGATION
    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir-c
  )
  # API libraries compile with hidden visibility and macros that enable
  # exporting from the DLL. Only apply to the obj lib, which only affects
  # the exports via a shared library.
  set_target_properties(obj.${name}
    PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
  target_compile_definitions(obj.${name}
    PRIVATE
    -DMLIR_CAPI_BUILDING_LIBRARY=1
  )
endfunction()

function(add_rocmlir_tool name)
  set(exclude_from_all "")
  if (BUILD_FAT_LIBROCKCOMPILER)
    set(exclude_from_all "EXCLUDE_FROM_ALL")
    # Temporarily disable "Building tools" to avoid generating install targets for
    # unbuilt files
    set(LLVM_BUILD_TOOLS OFF)
    set(EXCLUDE_FROM_ALL ON) # LLVM functions read this variable, set it paranoidly
  endif()
  add_mlir_tool(${name} ${exclude_from_all} ${ARGN})
endfunction()
