add_mlir_conversion_library(MLIRMHALToGPU
  MHALToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MHALToGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAsyncDialect
  MLIRMHAL
  MLIRGPUOps
  MLIRLLVMDialect
  MLIRTransforms
  )
