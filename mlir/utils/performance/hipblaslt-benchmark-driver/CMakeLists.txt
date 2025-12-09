# Find hipBLAS and hipBLASLt libraries directly
find_library(HIPBLAS_LIBRARY
  NAMES hipblas
  PATHS /opt/rocm/lib /opt/rocm/lib64
  PATH_SUFFIXES lib lib64
)

find_library(HIPBLASLT_LIBRARY
  NAMES hipblaslt
  PATHS /opt/rocm/lib /opt/rocm/lib64
  PATH_SUFFIXES lib lib64
)

# Check if libraries and include directories exist
if (HIPBLAS_LIBRARY AND HIPBLASLT_LIBRARY AND
    EXISTS "/opt/rocm/include/hipblas" AND EXISTS "/opt/rocm/include/hipblaslt")
  set(LIBS ${HIPBLAS_LIBRARY} ${HIPBLASLT_LIBRARY} benchmark-driver-utils)

  add_executable(hipblaslt-benchmark-driver
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
  message(STATUS "Could not find hipBLAS/hipBLASLt libraries or include directories, not building hipBLASLt benchmark driver")
endif()