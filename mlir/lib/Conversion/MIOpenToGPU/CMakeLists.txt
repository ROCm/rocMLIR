add_mlir_conversion_library(MLIRMIOpenToGPU
  MIOpenToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MIOpenToGPU

  DEPENDS
  MLIRConversionPassIncGen
  MLIRMIOpenConversionPassIncGen
)
target_link_libraries(MLIRMIOpenToGPU
  PUBLIC
  MLIRAffineToStandard
  MLIRMIOpenTransforms
  MLIRLLVMDialect
  MLIRIR
  MLIRPass
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRSupport
  MLIRTransforms
  MLIRVectorDialect
  LLVMCore
  LLVMSupport
  )
