add_rocmlir_dialect_library(MLIRMIGraphXPipeline
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
  MLIRRockOps
  MLIRRockTuning
  MLIRSCFToControlFlow
  MLIRSupport
  # MLIRTransformUtils
)
