add_mlir_dialect_library(MLIRArmNeonTransforms
<<<<<<< HEAD
  LowerContractionToNeonI8MMPattern.cpp
=======
  LowerContractToNeonPatterns.cpp
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

  DEPENDS
  MLIRArmNeonIncGen

  LINK_LIBS PUBLIC
  MLIRArmNeonDialect
  MLIRFuncDialect
  MLIRVectorDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
