file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRMIOpenOps
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/MIOpenOps
  )
add_dependencies(MLIRMIOpenOps MLIRMIOpenOpsIncGen MLIRStandardOps LLVMSupport)
target_link_libraries(MLIRMIOpenOps LLVMSupport)

add_subdirectory(CppOutput)
