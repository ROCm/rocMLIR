# TODO: factor out these properties into upstream/downstream
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_mlir_public_c_api_library(MLIRCAPIRegisterRocMLIR
  RegisterRocMLIR.cpp

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${conversion_libs}

  MLIRCAPIIR
)
