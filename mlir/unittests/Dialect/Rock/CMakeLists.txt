set(ROCK_UNITTEST_SOURCES
  TransformMapBuilderTests.cpp
  TosaUtilsTests.cpp
  loweringUtilsTests.cpp
  transformMapUtilsTests.cpp
  TuningParamSpaceTests.cpp
  GeneralGemmParamsAttrTests.cpp
  AccelGemmParamsAttrTests.cpp
  GemmGemmParamsAttrTests.cpp
  ParamLookupTableTests.cpp
)

if(NOT WIN32)
  list(APPEND ROCK_UNITTEST_SOURCES AmdArchDbTests.cpp)
endif()

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
