set(LLVM_OPTIONAL_SOURCES
  mixrir.c
  tosa_miir.cpp
)

set(LLVM_LINK_COMPONENTS
  )

add_llvm_executable(mlir-mixr-capi-test
  mixrir.c
  )
llvm_update_compile_flags(mlir-mixr-capi-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-mixr-capi-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegistration
  )

add_llvm_executable(mlir-tosa-miir-test
  tosa_miir.cpp
  )
llvm_update_compile_flags(mlir-tosa-miir-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-tosa-miir-test
  PRIVATE
  MLIRCAPIIR
  MLIRCAPIRegistration
  MLIRMIOpenPipeline
  LLVMAMDGPUAsmParser
  LLVMX86AsmParser
  )
