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

# `RocmRuntimeLoaderTests` exercises the cross-library coordination
# contract between `MLIRRocmRuntimeLoader` and `RocmSystemDetect`. The
# latter lives in `MLIRRocmExecutionEngineUtils`, which is only built
# as a standalone target in the standard shared-library configuration.
# In the fat-library configuration (`BUILD_FAT_LIBROCKCOMPILER=ON`),
# `RocmSystemDetect` is not bundled into `librockCompiler.a` -- it is
# only exposed to the rocMLIR JIT path -- so we omit the test sub-suite
# there. The contract being tested is identical in both configurations.
if (TARGET MLIRRocmExecutionEngineUtils)
  list(APPEND ROCK_UNITTEST_SOURCES RocmRuntimeLoaderTests.cpp)
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
  MLIRRocmRuntimeLoader
)
if (TARGET MLIRRocmExecutionEngineUtils)
  target_link_libraries(MLIRRockUnitTests PRIVATE MLIRRocmExecutionEngineUtils)
endif()
