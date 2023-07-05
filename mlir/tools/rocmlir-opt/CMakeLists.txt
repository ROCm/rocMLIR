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

set(disable_install "")
if (BUILD_FAT_LIBROCKCOMPILER)
  set(disable_install "DISABLE_INSTALL")
endif()

add_mlir_library(RocMLIROptMain
  rocmlir-opt.cpp

  EXCLUDE_FROM_LIBMLIR
  ${disable_install}

  LINK_LIBS PRIVATE
  ${LIBS}
  )
set_target_properties(RocMLIROptMain PROPERTIES EXCLUDE_FROM_ALL ${BUILD_FAT_LIBROCKCOMPILER})

add_rocmlir_tool(rocmlir-opt
  rocmlir-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(rocmlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(rocmlir-opt)

mlir_check_all_link_libraries(rocmlir-opt)
