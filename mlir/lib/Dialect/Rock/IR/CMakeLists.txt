
add_rocmlir_dialect_library(MLIRRockOps
  TransformMapBuilder.cpp
  RockDialect.cpp
  RockGemmWrapperInterface.cpp
  RockConvInterface.cpp
  RockTuningParamAttrInterface.cpp
  RockAccelTuningParamAttrInterface.cpp
  MfmaInsnGroup.cpp
  WmmaInsnGroup.cpp

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
  MLIRAMDGPUDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRDialect
  MLIRAnalysis
  MLIRParser
  MLIRROCDLDialect
  MLIRSupport
  )

