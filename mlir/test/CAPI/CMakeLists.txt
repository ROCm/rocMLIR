set(LLVM_OPTIONAL_SOURCES
  mixrir.c
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
  MLIRPublicAPI
  )
