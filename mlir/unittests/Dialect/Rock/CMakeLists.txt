set(ROCK_UNITTEST_SOURCES
  AmdArchDbTests.cpp
  TransformMapBuilderTests.cpp
  TosaUtilsTests.cpp
  loweringUtilsTests.cpp
  transformMapUtilsTests.cpp
  GeneralGemmParamsAttrTests.cpp
  AccelGemmParamsAttrTests.cpp
  GemmGemmParamsAttrTests.cpp
  ParamLookupTableTests.cpp
)

add_rocmlir_unittest(MLIRRockUnitTests
  PARTIAL_SOURCES_INTENDED
  ${ROCK_UNITTEST_SOURCES}
)

target_link_libraries(MLIRRockUnitTests
  PRIVATE
  MLIRRockOps
  MLIRRockUtility
  MLIRRockTuning
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRIR
)
