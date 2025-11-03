# Find hipBLAS and hipBLASLt libraries directly
find_library(HIPBLAS_LIBRARY
  NAMES hipblas
  PATHS /opt/rocm/lib /opt/rocm/lib64
  PATH_SUFFIXES lib lib64
  REQUIRED
)

find_library(HIPBLASLT_LIBRARY
  NAMES hipblaslt
  PATHS /opt/rocm/lib /opt/rocm/lib64
  PATH_SUFFIXES lib lib64
  REQUIRED
)

# Check if include directories exist
if (EXISTS "/opt/rocm/include/hipblas" AND EXISTS "/opt/rocm/include/hipblaslt")
  set(LIBS ${HIPBLAS_LIBRARY} ${HIPBLASLT_LIBRARY} benchmark-driver-utils)

  add_executable(hipblaslt-benchmark-driver
    EXCLUDE_FROM_ALL
    hipblaslt-benchmark-driver.cpp
  )

  target_link_libraries(hipblaslt-benchmark-driver PRIVATE ${LIBS})
  target_include_directories(hipblaslt-benchmark-driver PRIVATE 
    /opt/rocm/include
    /opt/rocm/include/hipblas
    /opt/rocm/include/hipblaslt
  )
  set_target_properties(hipblaslt-benchmark-driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  
  message(STATUS "Building hipBLASLt benchmark driver with libraries: ${LIBS}")
else()
  message(STATUS "Could not find hipBLAS/hipBLASLt include directories, not building hipBLASLt benchmark driver")
endif()
