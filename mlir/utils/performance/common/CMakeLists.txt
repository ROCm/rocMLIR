find_package(hip PATHS /opt/rocm)
if(NOT hip_FOUND)
  message(WARNING "Could not find HIP, not building benchmark utils")
endif()

if (hip_FOUND)
  set(EXCLUDE_FROM_ALL TRUE)
  add_library(benchmark-driver-utils benchmarkUtils.cpp)
  target_link_libraries(benchmark-driver-utils hip::device)
  set_target_properties(benchmark-driver-utils
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
endif()
