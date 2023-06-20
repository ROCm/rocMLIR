add_rocmlir_public_c_api_library(MLIRCAPIMIGraphX
  MIGraphX.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMIGraphX
  MLIRMIGraphXPipeline
  MLIRRockPipeline
)

add_rocmlir_public_c_api_library(MLIRCAPIRock
  Rock.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRRockOps
  MLIRRockTuning
)
