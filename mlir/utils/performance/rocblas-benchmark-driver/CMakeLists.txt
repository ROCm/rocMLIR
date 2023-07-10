find_package(rocblas CONFIG HINTS /opt/rocm)

if (rocblas_FOUND)
  set(LIBS ${rocblas_LIBRARY} benchmark-driver-utils)

  add_executable(rocblas-benchmark-driver
    EXCLUDE_FROM_ALL
    rocblas-benchmark-driver.cpp
  )

  target_link_libraries(rocblas-benchmark-driver PRIVATE ${LIBS})
  set_target_properties(rocblas-benchmark-driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
else()
  message(STATUS "Could not find rocBLAS, not building a rocBLAS benchmark driver")
endif()
