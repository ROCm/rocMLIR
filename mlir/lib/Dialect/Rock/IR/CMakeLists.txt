
add_rocmlir_dialect_library(MLIRRockOps
  TransformMapBuilder.cpp
  RockDialect.cpp
  RockGemmWrapperInterface.cpp
  RockConvInterface.cpp
  RockTuningParamAttrInterface.cpp
  RockAccelTuningParamAttrInterface.cpp
  MfmaInsnGroup.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rock

  DEPENDS
  MLIRRockAttrDefsIncGen
  MLIRRockGemmWrapperInterfaceIncGen
  MLIRRockTuningParamAttrInterfaceIncGen
  MLIRRockAccelTuningParamAttrInterfaceIncGen
  MLIRRockOpsIncGen
  )


target_link_libraries(MLIRRockOps
  PUBLIC
  MLIRAffineDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransformOps
  MLIRGPUOps
  MLIRAMDGPUDialect
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRDialect
  MLIRAnalysis
  MLIRParser
  MLIRROCDLDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTensorDialect
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRToLLVMIRTranslationRegistration
  MLIRTransforms
  LLVMSupport
  )

