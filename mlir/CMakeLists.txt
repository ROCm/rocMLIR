message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_BUILD_LIBRARY_DIR: ${LLVM_BUILD_LIBRARY_DIR}")
message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
check_c_compiler_flag("-Werror=implicit-function-declaration" C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION)
append_if(C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION "-Werror=implicit-function-declaration" CMAKE_C_FLAGS)

# Warn on undefined macros. This is often an indication that an include to
# `mlir-config.h` or similar is missing.
check_c_compiler_flag("-Wundef" C_SUPPORTS_WUNDEF)
append_if(C_SUPPORTS_WUNDEF "-Wundef" CMAKE_C_FLAGS)
append_if(C_SUPPORTS_WUNDEF "-Wundef" CMAKE_CXX_FLAGS)

# Forbid mismatch between declaration and definition for class vs struct. This is
# harmless on Unix systems, but it'll be a ticking bomb for MSVC/Windows systems
# where it creeps into the ABI.
check_c_compiler_flag("-Werror=mismatched-tags" C_SUPPORTS_WERROR_MISMATCHED_TAGS)
append_if(C_SUPPORTS_WERROR_MISMATCHED_TAGS "-Werror=mismatched-tags" CMAKE_C_FLAGS)
append_if(C_SUPPORTS_WERROR_MISMATCHED_TAGS "-Werror=mismatched-tags" CMAKE_CXX_FLAGS)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # Silence a false positive GCC -Wunused-but-set-parameter warning in
  # constexpr cases. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85827
  # for details
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
    check_cxx_compiler_flag("-Wno-unused-but-set-parameter" CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER)
    append_if(CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER "-Wno-unused-but-set-parameter" CMAKE_CXX_FLAGS)
  endif()
  # Silence a false positive GCC -Wdeprecated-copy warning in cases where
  # a copy operator is defined through "using" a base class copy operator.
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
    check_cxx_compiler_flag("-Wno-deprecated-copy" CXX_SUPPORTS_WNO_DEPRECTAED_COPY)
    append_if(CXX_SUPPORTS_WNO_DEPRECTAED_COPY "-Wno-deprecated-copy" CMAKE_CXX_FLAGS)
  endif()
endif()

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
# mlir-generic-headers are dialect-independent.
add_custom_target(mlir-generic-headers)
set_target_properties(mlir-generic-headers PROPERTIES FOLDER "MLIR/Resources")
# mlir-headers may be dialect-dependent.
add_custom_target(mlir-headers)
set_target_properties(mlir-headers PROPERTIES FOLDER "MLIR/Resources")
add_dependencies(mlir-headers mlir-generic-headers)
add_custom_target(mlir-doc)
set_target_properties(mlir-doc PROPERTIES FOLDER "MLIR/Docs")

# Only enable execution engine if the native target is available.
if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE 0)
endif()

# Build the ROCm conversions and run according tests if the AMDGPU backend
# is available.
if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_ROCM_CONVERSIONS 1)
else()
  set(MLIR_ENABLE_ROCM_CONVERSIONS 0)
endif()

set(MLIR_ENABLE_CUDA_RUNNER 0 CACHE BOOL "Enable building the MLIR CUDA runner")
set(MLIR_ENABLE_ROCM_RUNNER 0 CACHE BOOL "Enable building the MLIR ROCm runner")
set(MLIR_ENABLE_SYCL_RUNNER 0 CACHE BOOL "Enable building the MLIR SYCL runner")
set(MLIR_ENABLE_SPIRV_CPU_RUNNER 0 CACHE BOOL "Enable building the MLIR SPIR-V cpu runner")
set(MLIR_ENABLE_VULKAN_RUNNER 0 CACHE BOOL "Enable building the MLIR Vulkan runner")
set(MLIR_ENABLE_NVPTXCOMPILER 0 CACHE BOOL
    "Statically link the nvptxlibrary instead of calling ptxas as a subprocess \
    for compiling PTX to cubin")

set(MLIR_ENABLE_PDL_IN_PATTERNMATCH 1 CACHE BOOL "Enable PDL in PatternMatch")

option(MLIR_INCLUDE_TESTS
       "Generate build targets for the MLIR unit tests."
       ${LLVM_INCLUDE_TESTS})

option(MLIR_INCLUDE_INTEGRATION_TESTS
       "Generate build targets for the MLIR integration tests.")

set(MLIR_INSTALL_AGGREGATE_OBJECTS 1 CACHE BOOL
    "Installs object files needed for add_mlir_aggregate to work out of \
    tree. Package maintainers can disable this to exclude these assets if \
    not desired. Enabling this will result in object files being written \
    under lib/objects-{CMAKE_BUILD_TYPE}.")

set(MLIR_BUILD_MLIR_C_DYLIB 0 CACHE BOOL "Builds libMLIR-C shared library.")

set(MLIR_LINK_MLIR_DYLIB ${LLVM_LINK_LLVM_DYLIB} CACHE BOOL
    "Link tools against libMLIR.so")

configure_file(
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Config/mlir-config.h.cmake
  ${MLIR_INCLUDE_DIR}/mlir/Config/mlir-config.h)

#-------------------------------------------------------------------------------
# Python Bindings Configuration
# Requires:
#   The pybind11 library can be found (set with -DPYBIND_DIR=...)
#   The python executable is correct (set with -DPython3_EXECUTABLE=...)
# By default, find_package and probing for installed pybind11 is performed.
# Super projects can set MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES=ON to
# disable all package setup and control it themselves.
#-------------------------------------------------------------------------------

set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir"
  CACHE STRING "nanobind domain for MLIR python bindings.")
set(MLIR_ENABLE_BINDINGS_PYTHON 0 CACHE BOOL
       "Enables building of Python bindings.")
set(MLIR_DETECT_PYTHON_ENV_PRIME_SEARCH 1 CACHE BOOL
       "Prime the python detection by searching for a full 'Development' \
       component first (temporary while diagnosing environment specific Python \
       detection issues)")
set(MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES 0 CACHE BOOL
       "Performs python dev package configuration sufficient to use all MLIR \
       python features. Super-projects that wish to control their own setup \
       must perform an appropriate find_package of Python3 with \
       'Development.Module' and ensure that find_package(pybind11) is \
       satisfied (and keep up to date as requirements evolve).")

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  # Note that both upstream and downstreams often call this macro. It gates
  # internally on the MLIR_CONFIGURE_PYTHON_DEV_PACKAGES option.
  mlir_configure_python_dev_packages()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/mlir/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(LLVM_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(LLVM_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
set(ARG_PRODUCT_NAME ${CMAKE_PROJECT_NAME} CACHE STRING "")
set(ARG_VERSION_STRING ${CMAKE_PROJECT_VERSION} CACHE STRING "")

add_subdirectory(docs)
add_subdirectory(include/mlir)
add_subdirectory(lib)
# C API needs all dialects for registration, but should be built before tests.
add_subdirectory(lib/CAPI)

if (MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_custom_target(RocMLIRUnitTests)
  if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
  else()
    message(WARNING "gtest not found, unittests will not be available LLVM_MAIN_SRC_DIR=${LLVM_MAIN_SRC_DIR}")
  endif()
  add_subdirectory(test)
endif()
add_subdirectory(tools)
add_subdirectory(utils)
