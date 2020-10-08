add_mlir_conversion_library(MLIRMIOpenToGPU
  MIOpenToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MIOpenToGPU

  DEPENDS
  MLIRConversionPassIncGen
)
target_link_libraries(MLIRMIOpenToGPU
  PUBLIC
  MLIRAffineToStandard
  MLIRMIOpenTransforms
  MLIRGPU
  MLIRLLVMIR
  MLIRIR
  MLIRPass
  MLIRSCFToStandard
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTransforms
  MLIRVector
  LLVMCore
  LLVMSupport
  )
