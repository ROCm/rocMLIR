add_mlir_dialect_library(MLIRMHALTransforms
  AnnotateAccessKinds.cpp
  Bufferize.cpp
  BufferizableOpInterfaceImpl.cpp
  DropMetadata.cpp
  EmulateNarrowType.cpp
  InferGraph.cpp
  PackageTargets.cpp
  SelectTargets.cpp
  TargetKernels.cpp
  Prefill.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MHAL

  DEPENDS
  MLIRMHALPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithTransforms
  MLIRDialect
  MLIRMHAL
  MLIRMHALSupport
  MLIRFuncDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRMemRefTransforms
  MLIRPass
  MLIRLLVMDialect
  MLIRSupport
  MLIRTransformUtils
)
