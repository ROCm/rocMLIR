# Performance tests can't meaningfully be used without the ability to
# run code we generate from within our build, which is controlled by
# MLIR_ENABLE_ROCM_RUNNNER . Furthermore, not guarding the performance
# subdirectory leads to static library builds trying to find HIP, which
# has caused fatal CMake errors in cases where ROCm is only partially
# installed.
if (MLIR_ENABLE_ROCM_RUNNER)
  add_subdirectory(performance)
endif()
