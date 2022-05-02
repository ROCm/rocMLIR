add_mlir_dialect_library(MLIRMIGraphXPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFunc
  LLVMROCmBackendUtils
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
