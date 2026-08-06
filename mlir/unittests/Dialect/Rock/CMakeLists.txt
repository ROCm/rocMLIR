set(ROCK_UNITTEST_SOURCES
  TransformMapBuilderTests.cpp
  TosaUtilsTests.cpp
  loweringUtilsTests.cpp
  transformMapUtilsTests.cpp
  GeneralGemmParamsAttrTests.cpp
  AccelGemmParamsAttrTests.cpp
  GemmGemmParamsAttrTests.cpp
  ParamLookupTableTests.cpp
  RockTuningTests.cpp
)

# AmdArchDbTests directly includes hip/hip_runtime_api.h and exercises
# lookupArchInfo("native:N"), so it can only be built when the native-arch
# path is compiled in. Skip on Windows (no HIP/HSA) and whenever
# ROCMLIR_ENABLE_NATIVE_ARCH is OFF (e.g. BUILD_FAT_LIBROCKCOMPILER).
if(NOT WIN32 AND ROCMLIR_ENABLE_NATIVE_ARCH)
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
