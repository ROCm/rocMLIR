add_rocmlir_conversion_library(MLIRRockToGPU
  RockToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/RockToGPU

  DEPENDS
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen
  MLIRRockPassIncGen
)
target_link_libraries(MLIRRockToGPU
  PUBLIC
  MLIRRockTransforms
  MLIRAMDGPUDialect
  MLIRDLTIDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRVectorDialect
  LLVMCore
  LLVMSupport
  )
