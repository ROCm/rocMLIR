set(LLVM_OPTIONAL_SOURCES
  mixrir.c
  tosa_miir.cpp
  mixr_cobj.cpp
  mixr_full.c
)

set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD})

add_llvm_executable(mlir-mixr-capi-test
  mixrir.c
)
llvm_update_compile_flags(mlir-mixr-capi-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-mixr-capi-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegisterRocMLIR
)

add_llvm_executable(mlir-tosa-miir-test
  tosa_miir.cpp
)
llvm_update_compile_flags(mlir-tosa-miir-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-tosa-miir-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegisterRocMLIR
  MLIRRockPipeline
)

add_llvm_executable(mlir-mixr-full-test
  mixr_cobj.cpp
  )
llvm_update_compile_flags(mlir-mixr-full-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-mixr-full-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegisterRocMLIR
  MLIRCAPIMIGraphX
  MLIRCAPIGPU
  MLIRMIGraphXPipeline
  MLIRRockPipeline
)

add_llvm_executable(mlir-mixr-fullc-test
  mixr_full.c
  )
llvm_update_compile_flags(mlir-mixr-fullc-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-mixr-fullc-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegisterRocMLIR
  MLIRCAPIMIGraphX
  MLIRCAPIRock
  MLIRCAPIGPU
)
