add_mlir_dialect_library(MLIRLLVMIRTransforms
  DIScopeForLLVMFuncOp.cpp
  LegalizeForExport.cpp
  OptimizeForNVVM.cpp
  RequestCWrappers.cpp
  TypeConsistency.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRLLVMCommonConversion
  MLIRTransforms
  MLIRNVVMDialect
  )
