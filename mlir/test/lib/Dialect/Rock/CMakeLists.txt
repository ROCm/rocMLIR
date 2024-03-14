add_rocmlir_test_library(MLIRRockTestPasses
  TestCollapseContiguousMerges.cpp
  TestVectorizationInference.cpp
  TestRockMultibuffer.cpp
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRRockOps
  MLIRRockTransforms
  MLIRRockUtility
  MLIRPass)
