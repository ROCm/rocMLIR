add_rocmlir_test_library(MLIRRockTestPasses
  TestCollapseContiguousMerges.cpp
  TestVectorizationInference.cpp
  TestRockMultibuffer.cpp
  TestBufferDependencyAnalysis.cpp
  TestFunctionFusibility.cpp
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRRockAnalysis
  MLIRRockOps
  MLIRRockTransforms
  MLIRRockUtility
  MLIRPass)
