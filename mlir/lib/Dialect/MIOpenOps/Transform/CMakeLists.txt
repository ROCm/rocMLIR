add_llvm_library(MLIRMIOpenTransforms
  AffineTransforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
)

add_dependencies(MLIRMIOpenTransforms
  MLIRMIOpenOpsIncGen
  MLIRStandardOps
  )

target_link_libraries(MLIRMIOpenTransforms
  MLIRAnalysis
  MLIRMIOpenOps
  MLIRPass
  MLIRTransformUtils
  )
