add_rocmlir_dialect_library(MLIRRockTuning
  GeneralGemmBlockStructure.cpp
  ConvContext.cpp
  GridwiseGemmParams.cpp
  GridwiseGemmGemmParams.cpp
  RockTuningImpl.cpp
  ParamLookupTable.cpp
  QuickTuningClassifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock/Tuning

  DEPENDS
  MLIRRockOps
  MLIRSupport
  MLIRRockPassIncGen
)

find_package(xgboost REQUIRED)

target_link_libraries(MLIRRockTuning
  PRIVATE
  MLIRRockUtility
  MLIRIR
  xgboost::xgboost
)

if(EXISTS ${CMAKE_SOURCE_DIR}/models)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/
          DESTINATION bin/models
          FILES_MATCHING PATTERN "*.ubj")
endif()
