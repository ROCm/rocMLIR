add_mlir_conversion_library(MLIRRockToGPU
  RockToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/RockToGPU

  DEPENDS
  MLIRConversionPassIncGen
  MLIRRockConversionPassIncGen
  MLIRRockPassIncGen
)
target_link_libraries(MLIRRockToGPU
  PUBLIC
  MLIRAffineToStandard
  MLIRRockTransforms
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
