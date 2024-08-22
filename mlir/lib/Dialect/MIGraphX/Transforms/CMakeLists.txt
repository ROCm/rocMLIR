add_rocmlir_dialect_library(MLIRMIGraphXTransforms
  MIGraphXTransform.cpp
  RealizeInt4.cpp

  DEPENDS
  MLIRMIGraphXPassIncGen

  LINK_LIBS PUBLIC
  MLIRMIGraphXDialect
  MLIRPass
  MLIRFuncDialect
  MLIRTransforms
  MLIRSupport
)
