set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_test_libs}
  MLIRLspServerLib
  )

add_llvm_tool(rocmlir-lsp-server
  rocmlir-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(rocmlir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(rocmlir-lsp-server)

mlir_check_all_link_libraries(rocmlir-lsp-server)
