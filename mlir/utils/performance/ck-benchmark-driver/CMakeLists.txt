find_package(composable_kernel CONFIG 1.0.0 COMPONENTS device_operations)
if(NOT composable_kernel_FOUND)
  message(WARNING "Could not find composable-kernels, not building a CK benchmark driver")
endif()

if (composable_kernel_FOUND)
  find_package(hip REQUIRED PATHS /opt/rocm)

  set(LLVM_OPTIONAL_SOURCES
    null.cpp)

  set(LLVM_LINK_COMPONENTS
    Support)

  set(LIBS composable_kernel::device_operations hip::device benchmark-driver-utils)

  set(EXCLUDE_FROM_ALL TRUE)
  add_llvm_executable(ck-benchmark-driver

    ck-benchmark-driver.cpp
    DEPENDS ${LIBS})

  target_link_libraries(ck-benchmark-driver PRIVATE ${LIBS})
  llvm_update_compile_flags(ck-benchmark-driver)
  mlir_check_all_link_libraries(ck-benchmark-driver)
endif()
