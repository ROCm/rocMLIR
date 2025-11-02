set(ROCK_UNITTEST_SOURCES
  TransformMapBuilderTests.cpp
  TosaUtilsTests.cpp
  loweringUtilsTests.cpp
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
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRIR
)
