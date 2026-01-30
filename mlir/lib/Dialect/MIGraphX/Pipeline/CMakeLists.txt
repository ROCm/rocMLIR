add_rocmlir_dialect_library(MLIRMIGraphXPipeline
  Pipeline.cpp

  LINK_LIBS PUBLIC
  MLIRGPUToROCDLTransforms
  MLIRMIGraphXToTosa
  MLIRMIGraphXToLinalg
  MLIRMIGraphXTransforms
  MLIRRockOps
  MLIRRockTuning
)
