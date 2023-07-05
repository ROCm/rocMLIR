set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  MLIRLspServerLib
  MLIRMHAL
  MLIRMHALTransforms
  )

add_rocmlir_tool(rocmlir-lsp-server
  rocmlir-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(rocmlir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(rocmlir-lsp-server)

mlir_check_all_link_libraries(rocmlir-lsp-server)
