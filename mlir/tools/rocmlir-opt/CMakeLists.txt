set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)
get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )


set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  ${rocmlir_test_libs}
  MLIROptLib
  MLIRMHAL
  MLIRMHALTransforms
  MLIRMHALPipeline
  )

# Exclude from libMLIR.so because this has static options intended for
# opt-like tools only.
add_mlir_library(RocMLIROptMain
  rocmlir-opt.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LIBS}
  )

add_llvm_tool(rocmlir-opt
  rocmlir-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(rocmlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(rocmlir-opt)

mlir_check_all_link_libraries(rocmlir-opt)
