add_llvm_library(MLIRMIOpenTransforms
  AffixTuningParameters.cpp
  AffineTransforms.cpp
  ConvertMIOpenOpsToStd.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
)

add_dependencies(MLIRMIOpenTransforms
  MLIRMIOpenOpsIncGen
  MLIRStandardOps
  MLIRLoopToStandard
  )

target_link_libraries(MLIRMIOpenTransforms
  MLIRAnalysis
  MLIRMIOpenOps
  MLIRPass
  MLIRTransformUtils
  )
