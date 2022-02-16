add_mlir_dialect_library(MLIRMIGraphXPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  LLVMROCmBackendUtils
  MLIRGPUToROCDLTransforms
  MLIRMIGraphXToTosa
  MLIRGPUToMIGraphX
  MLIRIR
  MLIRPass
  MLIRLLVMIR
  MLIRStandard
  MLIRAffineToStandard
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)
