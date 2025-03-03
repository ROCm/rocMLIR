add_rocmlir_test_library(MLIRMHALTestPasses
  TestTosaPartition.cpp
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRMHALTransforms
  )
