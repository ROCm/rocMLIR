#add_subdirectory(mlir-cuda-runner)
#add_subdirectory(mlir-cpu-runner)
add_subdirectory(mlir-opt)
#add_subdirectory(mlir-reduce)
#add_subdirectory(mlir-rocm-runner)
#add_subdirectory(mlir-shlib)
#add_subdirectory(mlir-spirv-cpu-runner)
add_subdirectory(mlir-translate)
#add_subdirectory(mlir-vulkan-runner)

# MIOpen dialect specific.
add_subdirectory(mlir-miopen-driver)
add_subdirectory(mlir-miopen-lib)
