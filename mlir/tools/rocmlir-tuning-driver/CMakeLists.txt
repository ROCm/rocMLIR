set(LLVM_OPTIONAL_SOURCES
  rocmlir-tuning-driver.cpp
  )

if(MLIR_ENABLE_ROCM_RUNNER)

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
)

# Grab HIP again, since we'll be using it directly
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
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/hip")
find_package(hip REQUIRED)
set(CMAKE_PREFIX_PATH "${REAL_CMAKE_PREFIX_PATH}")

# Supress compiler warnings from HIP headers
check_cxx_compiler_flag(-Wno-c++98-compat-extra-semi
  CXX_SUPPORTS_NO_CXX98_COMPAT_EXTRA_SEMI_FLAG)
if (CXX_SUPPORTS_CXX98_COMPAT_EXTRA_SEMI_FLAG)
  target_compile_options(mlir_rocm_runtime PRIVATE
    "-Wno-c++98-compat-extra-semi")
endif()
check_cxx_compiler_flag(-Wno-return-type-c-linkage
    CXX_SUPPORTS_WNO_RETURN_TYPE_C_LINKAGE_FLAG)
if (CXX_SUPPORTS_WNO_RETURN_TYPE_C_LINKAGE_FLAG)
  target_compile_options(mlir_rocm_runtime PRIVATE
    "-Wno-return-type-c-linkage")
endif()
check_cxx_compiler_flag(-Wno-nested-anon-types
  CXX_SUPPORTS_WNO_NESTED_ANON_TYPES_FLAG)
if (CXX_SUPPORTS_WNO_NESTED_ANON_TYPES_FLAG)
  target_compile_options(mlir_rocm_runtime PRIVATE
    "-Wno-nested-anon-types")
endif()
check_cxx_compiler_flag(-Wno-gnu-anonymous-struct
  CXX_SUPPORTS_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
if (CXX_SUPPORTS_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
  target_compile_options(mlir_rocm_runtime PRIVATE
    "-Wno-gnu-anonymous-struct")
endif()

set_property(TARGET rocmlir-tuning-driver
  PROPERTY INSTALL_RPATH_USE_LINK_PATH ON)

target_link_libraries(rocmlir-tuning-driver PRIVATE ${LIBS})
target_link_libraries(rocmlir-tuning-driver PUBLIC benchmark-driver-utils)
target_link_libraries(rocmlir-tuning-driver PUBLIC hip::host hip::amdhip64)
llvm_update_compile_flags(rocmlir-tuning-driver)
endif()
