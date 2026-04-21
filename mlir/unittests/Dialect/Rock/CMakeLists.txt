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

# The native-arch tests load `mlir_rocm_arch_runtime` at run time via a
# private `dlmopen` / `LoadLibraryW` call. Make sure it is built and placed
# alongside the test binary when HIP is available.
if (TARGET mlir_rocm_arch_runtime)
  add_dependencies(MLIRRockUnitTests mlir_rocm_arch_runtime)
endif()
