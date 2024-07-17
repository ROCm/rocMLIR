add_rocmlir_dialect_library(MLIRMIGraphXTransforms
  MIGraphXTransform.cpp
  IREEGen.cpp

  DEPENDS
  MLIRMIGraphXPassIncGen

  LINK_LIBS PUBLIC
  MLIRMIGraphXDialect
  MLIRPass
  MLIRFuncDialect
  MLIRTransforms
  MLIRSupport
)
