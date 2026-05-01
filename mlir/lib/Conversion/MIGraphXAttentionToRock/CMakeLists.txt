add_rocmlir_conversion_library(MLIRMIGraphXAttentionToRock
  MIGraphXAttentionToRock.cpp

  DEPENDS
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMIGraphXDialect
  MLIRMemRefDialect
  MLIRRockOps
  MLIRPass
  MLIRTensorDialect
  MLIRBufferizationDialect
  MLIRTransformUtils
  MLIRSupport
  )
