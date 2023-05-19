add_mlir_dialect_library(MLIRMHALTransforms
  Bufferize.cpp
  BufferizableOpInterfaceImpl.cpp
  InferGraph.cpp
  PackageTargets.cpp
  SelectTargets.cpp
  TargetKernels.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MHAL

  DEPENDS
  MLIRMHALPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithTransforms
  MLIRDialect
  MLIRMHAL
  MLIRMHALSupport
  MLIRFuncDialect
  MLIRExecutionEngineUtils
  MLIRGPUTransforms
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)

