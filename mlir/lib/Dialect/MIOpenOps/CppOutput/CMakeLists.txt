add_llvm_library(MLIRMIOpenCpp
  ConvertToMIOpenCPP.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpenOps
  )
target_link_libraries(MLIRMIOpenCpp
  MLIRIR
  MLIRMIOpenOps
  MLIRStandardOps
  MLIRTranslation)

