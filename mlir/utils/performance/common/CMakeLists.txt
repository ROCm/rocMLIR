find_package(hip PATHS /opt/rocm)
if(NOT hip_FOUND)
  message(WARNING "Could not find HIP, not building benchmark utils")
endif()

if (hip_FOUND)
  set(LLVM_OPTIONAL_SOURCES
    null.cpp)

  set(LLVM_LINK_COMPONENTS
    Support)

  set(EXCLUDE_FROM_ALL TRUE)

  llvm_add_library(benchmark-driver-utils
    benchmarkUtils.cpp

    LINK_LIBS
    hip::device
  )

  llvm_update_compile_flags(benchmark-driver-utils)
  mlir_check_all_link_libraries(benchmark-driver-utils)
endif()
