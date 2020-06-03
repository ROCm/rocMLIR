set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIRMIOpenOps
  MLIRParser
  MLIRPass
  MLIRQuant
  MLIRShape
  MLIRSDBM
  MLIRStandardOps
  MLIRTransforms
  MLIRSupport
  MLIRIR
)

add_llvm_executable(mlir-miopen-driver
  mlir-miopen-driver.cpp

  DEPENDS
  ${LIBS}
)
llvm_update_compile_flags(mlir-miopen-driver)
target_link_libraries(mlir-miopen-driver PRIVATE ${LIBS})

mlir_check_link_libraries(mlir-miopen-driver)
