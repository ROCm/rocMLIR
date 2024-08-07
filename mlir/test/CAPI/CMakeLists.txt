set(LLVM_OPTIONAL_SOURCES
  mixrir.c
  tosa_miir.cc
  mixr_cobj.cc
  mixr_split_k.cpp
  mixr_full.c
)

set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD})

add_llvm_executable(mlir-mixr-capi-test
  EXCLUDE_FROM_ALL
  mixrir.c
)
llvm_update_compile_flags(mlir-mixr-capi-test)

target_link_libraries(mlir-mixr-capi-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIMIGraphX
  MLIRCAPIRegisterRocMLIR
)

# NOTE: This test is being archived and it might be fully removed in the future.
# add_llvm_executable(mlir-tosa-miir-test
#   EXCLUDE_FROM_ALL
#   tosa_miir.cc
# )
# llvm_update_compile_flags(mlir-tosa-miir-test)

# target_link_libraries(mlir-tosa-miir-test
#   PRIVATE
#   MLIRCAPIIR
#   MLIRCAPIRegisterRocMLIR
#   MLIRRockPipeline
# )

# NOTE: This test is being archived and it might be fully removed in the future.
# add_llvm_executable(mlir-mixr-full-test
#   EXCLUDE_FROM_ALL
#   mixr_cobj.cc
#   )
# llvm_update_compile_flags(mlir-mixr-full-test)

# target_link_libraries(mlir-mixr-full-test
#   PRIVATE
#   MLIRCAPIIR
#   MLIRCAPIRegisterRocMLIR
#   MLIRCAPIMIGraphX
#   MLIRCAPIGPU
#   MLIRMIGraphXPipeline
#   MLIRRockPipeline
# )

add_llvm_executable(mlir-mixr-fullc-test
  EXCLUDE_FROM_ALL
  mixr_full.c
  )
llvm_update_compile_flags(mlir-mixr-fullc-test)

target_link_libraries(mlir-mixr-fullc-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegisterRocMLIR
  MLIRCAPIMIGraphX
  MLIRCAPIRock
  MLIRCAPIGPU
)

add_llvm_executable(mlir-mixr-split-k-test
  EXCLUDE_FROM_ALL
  mixr_split_k.cpp
  )
llvm_update_compile_flags(mlir-mixr-split-k-test)

target_link_libraries(mlir-mixr-split-k-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegisterRocMLIR
  MLIRCAPIMIGraphX
  MLIRCAPIRock
  MLIRCAPIGPU
  MLIRMIGraphXPipeline
  MLIRRockPipeline
)
