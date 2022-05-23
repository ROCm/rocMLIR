add_mlir_dialect_library(MLIRLLVMIRTransforms
  LegalizeForExport.cpp
  SoftwareBf16.cpp

  DEPENDS
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRLLVMCommonConversion
  )
