add_llvm_library(MLIRMIOpenOpsToLLVM
  ConvertMIOpenOpsToStd.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpenOps
)
add_dependencies(
  MLIRMIOpenOpsToLLVM

  MLIRLoopToStandard
  MLIRLLVMIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIRMIOpenOpsToLLVM

  MLIRLoopToStandard
  MLIRLLVMIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
)

