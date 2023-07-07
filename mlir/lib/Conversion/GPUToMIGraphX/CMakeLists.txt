add_rocmlir_conversion_library(MLIRGPUToMIGraphX
  GPUToMIGraphX.cpp
  GPUToMIGraphXPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  RocMLIRConversionPassIncGen
  MLIRMIGraphXTypeIncGen
)

target_link_libraries(MLIRGPUToMIGraphX
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRGPUDialect
  MLIRMIGraphX
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUToROCDLTransforms
)
