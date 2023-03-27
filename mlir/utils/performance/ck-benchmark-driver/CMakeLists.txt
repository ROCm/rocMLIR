find_package(composable_kernel CONFIG 1.0.0 COMPONENTS device_operations)
if (composable_kernel_FOUND)
  find_package(hip REQUIRED PATHS /opt/rocm)
  set(LIBS composable_kernel::device_operations hip::device benchmark-driver-utils)
  set(EXCLUDE_FROM_ALL TRUE)

  add_executable(ck-benchmark-driver ck-benchmark-driver.cpp)
  target_link_libraries(ck-benchmark-driver PRIVATE ${LIBS})
  set_target_properties(ck-benchmark-driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
else()
  message(WARNING "Could not find composable-kernels, not building a CK benchmark driver")
endif()
