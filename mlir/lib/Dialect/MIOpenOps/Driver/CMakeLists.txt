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
whole_archive_link(mlir-miopen-driver ${LIBS})
target_link_libraries(mlir-miopen-driver PRIVATE MLIRIR ${LIBS} LLVMSupport)
