# TODO: Make the check source feature optional as an argument on *_add_library.
set(LLVM_OPTIONAL_SOURCES
  MIGraphX.cpp
)

add_mlir_public_c_api_library(MLIRCAPIMIGraphX
  MIGraphX.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRMIGraphX

)
