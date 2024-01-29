add_rocmlir_dialect_library(MLIRRockTuning
  GeneralGemmBlockStructure.cpp
  ConvContext.cpp
  GridwiseGemmParams.cpp
  RockTuningImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock/Tuning

  DEPENDS
  MLIRRockOps
  MLIRSupport
  MLIRRockPassIncGen
)

target_link_libraries(MLIRRockTuning
  PRIVATE
  MLIRRockUtility
  MLIRIR
)
