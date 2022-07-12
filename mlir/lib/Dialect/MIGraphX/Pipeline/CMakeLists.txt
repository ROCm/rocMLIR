add_mlir_dialect_library(MLIRMIGraphXPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFuncDialect
  MLIRGPUToROCDLTransforms
  MLIRMIGraphXToTosa
  MLIRGPUToMIGraphX
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)
