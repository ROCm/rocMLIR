set(ROCMLIR_UNITTEST_SOURCES
  TransformMapBuilderTests.cpp
  TosaUtilsTests.cpp
)

if(ROCMLIR_USE_ARCH_DB)
  list(APPEND ROCMLIR_UNITTEST_SOURCES AmdArchDbTests.cpp)
endif()

add_rocmlir_unittest(MLIRRockUnitTests
  PARTIAL_SOURCES_INTENDED
  ${ROCMLIR_UNITTEST_SOURCES}
)

target_link_libraries(MLIRRockUnitTests
  PRIVATE
  MLIRRockOps
  MLIRRockUtility
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRIR
)
