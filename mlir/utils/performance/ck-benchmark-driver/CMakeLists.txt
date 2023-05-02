find_package(composable_kernel CONFIG 1.0.0 COMPONENTS device_operations)
if(NOT composable_kernel_FOUND)
  message(WARNING "Could not find composable-kernels, not building a CK benchmark driver")
endif()

if (composable_kernel_FOUND)
  find_package(hip REQUIRED PATHS /opt/rocm)
  FILE(GLOB Int8Instances int8_int8_int32_instances/*.cpp)
  set(LIBS composable_kernel::device_operations hip::device benchmark-driver-utils)
  set(EXCLUDE_FROM_ALL TRUE)

  add_executable(ck-benchmark-driver ck-benchmark-driver.cpp ${Int8Instances})
  target_link_libraries(ck-benchmark-driver PRIVATE ${LIBS})
  set_target_properties(ck-benchmark-driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endif()
