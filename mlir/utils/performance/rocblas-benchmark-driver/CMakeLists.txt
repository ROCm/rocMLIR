find_package(rocblas CONFIG HINTS /opt/rocm)
if(NOT rocblas_FOUND)
  message(WARNING "Could not find rocBLAS, not building a rocBLAS benchmark driver")
endif()

if (rocblas_FOUND)
  set(LLVM_OPTIONAL_SOURCES
    null.cpp)

  set(LLVM_LINK_COMPONENTS
    Support)

  set(LIBS ${rocblas_LIBRARY})

  set(EXCLUDE_FROM_ALL TRUE)
  add_llvm_executable(rocblas-benchmark-driver

    rocblas-benchmark-driver.cpp
    ../common/benchmarkUtils.cpp
    DEPENDS ${LIBS})

  target_link_libraries(rocblas-benchmark-driver PRIVATE ${LIBS})
  llvm_update_compile_flags(rocblas-benchmark-driver)
  mlir_check_all_link_libraries(rocblas-benchmark-driver)
endif()
