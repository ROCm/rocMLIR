add_mlir_conversion_library(MLIRMHALToCPU
  MHALToCPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MHAL_MAIN_INCLUDE_DIR}/mlir/Conversion/MHALToCPU

  DEPENDS
  MHALConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRMHAL
  MLIRLLVMDialect
  MLIRTransforms
  )
