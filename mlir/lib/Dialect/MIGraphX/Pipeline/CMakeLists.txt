add_mlir_dialect_library(MLIRMIGraphXPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFunc
  MLIRGPUToROCDLTransforms
  MLIRMIGraphXToTosa
  MLIRGPUToMIGraphX
  MLIRIR
  MLIRPass
  MLIRLLVMIR
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)
