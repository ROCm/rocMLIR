add_rocmlir_conversion_library(MLIRMIGraphXAttentionToRock
  MIGraphXAttentionToRock.cpp

  DEPENDS
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMIGraphXDialect
  MLIRMemRefDialect
  MLIRRockOps
  MLIRPass
  MLIRTensorDialect
  MLIRBufferizationDialect
  MLIRTransformUtils
  MLIRSupport
  )
