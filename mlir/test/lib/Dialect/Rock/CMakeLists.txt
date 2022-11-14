add_rocmlir_test_library(MLIRRockTestPasses
  TestVectorizationInference.cpp
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRRockOps
  MLIRRockUtility
  MLIRPass)
