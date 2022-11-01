add_mlir_conversion_library(MLIRXModelToGPU
  XModelToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/XModelToGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAsyncDialect
  MLIRXModel
  MLIRGPUOps
  MLIRLLVMDialect
  MLIRTransforms
  )
