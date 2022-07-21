add_mlir_library(MLIRMIOpenTestPasses
  TestVectorizationInference.cpp
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRMIOpenOps
  MLIRMIOpenUtility
  MLIRPass)
