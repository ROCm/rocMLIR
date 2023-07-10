find_package(composable_kernel 1.0.0 COMPONENTS device_operations CONFIG)

if (composable_kernel_FOUND)
  find_package(hip REQUIRED PATHS /opt/rocm)
  set(LIBS composable_kernel::device_operations hip::device benchmark-driver-utils)

  add_executable(ck-benchmark-driver EXCLUDE_FROM_ALL ck-benchmark-driver.cpp)
  target_link_libraries(ck-benchmark-driver PRIVATE ${LIBS})
  set_target_properties(ck-benchmark-driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
else()
  message(STATUS "Could not find composable-kernels, not building a CK benchmark driver")
endif()
