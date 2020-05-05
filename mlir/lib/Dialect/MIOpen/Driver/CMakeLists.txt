set(LIBS
  MLIRAnalysis
  MLIRMIOpenOps
  MLIRParser
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  MLIRSupport
)

add_llvm_tool(mlir-miopen-driver
  mlir-miopen-driver.cpp
)
llvm_update_compile_flags(mlir-miopen-driver)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-miopen-driver PRIVATE
  ${dialect_libs}
  MLIRIR
  ${LIBS}
  LLVMSupport)
