find_package(composable_kernel 1.1.0 COMPONENTS device_gemm_operations CONFIG)

if (composable_kernel_FOUND)
  message(STATUS "Found Composable Kernel Library")
  find_package(hip REQUIRED PATHS /opt/rocm)
  set(LIBS composable_kernel::device_gemm_operations hip::device benchmark-driver-utils)

  add_library(ck-common INTERFACE IMPORTED)
  target_link_libraries(ck-common INTERFACE ${LIBS})
  target_compile_options(ck-common INTERFACE "-std=c++17")

  add_executable(ck-gemm-benchmark-driver EXCLUDE_FROM_ALL ck-gemm-benchmark-driver.cpp)
  target_link_libraries(ck-gemm-benchmark-driver PRIVATE ck-common)
  set_target_properties(ck-gemm-benchmark-driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  include(FetchContent)
  FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11
    GIT_TAG        v2.2.0
  )
  FetchContent_MakeAvailable(cli11)

  add_executable(ck-atten-benchmark-driver EXCLUDE_FROM_ALL ck-atten-benchmark-driver.cpp)
  target_link_libraries(ck-atten-benchmark-driver PRIVATE ck-common CLI11::CLI11)
  target_compile_options(ck-atten-benchmark-driver PRIVATE -Wno-covered-switch-default)
  set_target_properties(ck-atten-benchmark-driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  add_custom_target(ck-benchmark-driver)
  add_dependencies(ck-benchmark-driver ck-gemm-benchmark-driver ck-atten-benchmark-driver)
else()
  message(FATAL_ERROR "Could not find composable-kernels, not building a CK benchmark driver")
endif()
