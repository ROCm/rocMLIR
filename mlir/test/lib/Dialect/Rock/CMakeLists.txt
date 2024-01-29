add_rocmlir_test_library(MLIRRockTestPasses
  TestVectorizationInference.cpp
  TestRockMultibuffer.cpp
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRRockOps
  MLIRRockTransforms
  MLIRRockUtility
  MLIRPass)
