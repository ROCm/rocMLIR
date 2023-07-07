add_mlir_conversion_library(MLIRMHALToGPU
  MHALToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MHAL_MAIN_INCLUDE_DIR}/mlir/Conversion/MHALToGPU

  DEPENDS
  MHALConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRMHAL
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRTransforms
  )
