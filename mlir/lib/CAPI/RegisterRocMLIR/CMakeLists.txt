# TODO: factor out these properties into upstream/downstream
get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)
add_mlir_public_c_api_library(MLIRCAPIRegisterRocMLIR
  RegisterRocMLIR.cpp

  LINK_LIBS PUBLIC
  ${rocmlir_dialect_libs}

  MLIRCAPIIR
)
