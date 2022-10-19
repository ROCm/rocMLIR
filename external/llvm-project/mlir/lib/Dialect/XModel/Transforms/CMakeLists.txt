add_mlir_dialect_library(MLIRXModelTransforms
  AsyncGraph.cpp
  PackageTargets.cpp
  SelectTargets.cpp
  TargetKernels.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/XModel

  DEPENDS
  MLIRXModelPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticTransforms
  MLIRDialect
  MLIRXModel
  MLIRFuncDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
)

