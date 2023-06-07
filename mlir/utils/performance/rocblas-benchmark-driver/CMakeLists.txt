if(MLIR_USE_ROCBLAS)
  find_package(rocblas CONFIG HINTS /opt/rocm)
  if(NOT rocblas_FOUND)
    message(WARNING "Could not find rocBLAS, not building a rocBLAS benchmark driver")
  endif()

  if (rocblas_FOUND)
    set(LIBS ${rocblas_LIBRARY} benchmark-driver-utils)

    set(EXCLUDE_FROM_ALL TRUE)
    add_executable(rocblas-benchmark-driver
      rocblas-benchmark-driver.cpp
    )

    target_link_libraries(rocblas-benchmark-driver PRIVATE ${LIBS})
    set_target_properties(rocblas-benchmark-driver
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
  endif()
endif()
