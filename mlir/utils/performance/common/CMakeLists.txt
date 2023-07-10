find_package(hip PATHS /opt/rocm)

if (hip_FOUND)
  add_library(benchmark-driver-utils EXCLUDE_FROM_ALL benchmarkUtils.cpp)
  target_link_libraries(benchmark-driver-utils PUBLIC hip::host hip::amdhip64)
  set_target_properties(benchmark-driver-utils
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
else()
  message(STATUS "Could not find HIP, not building benchmark utils")
endif()
