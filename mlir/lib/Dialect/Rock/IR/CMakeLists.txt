
add_mlir_dialect_library(MLIRRockOps
  TransformMapBuilder.cpp
  RockDialect.cpp
  RockGemmWrapperInterface.cpp
  RockConvInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rock

  DEPENDS
  MLIRRockAttrDefsIncGen
  MLIRRockGemmWrapperInterfaceIncGen
  MLIRRockOpsIncGen
  )

target_link_libraries(MLIRRockOps
  PUBLIC
  MLIRGPUOps
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRDialect
  MLIRJitRunner
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRParser
  MLIRROCDLDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MLIRTransforms
  LLVMSupport
  )

