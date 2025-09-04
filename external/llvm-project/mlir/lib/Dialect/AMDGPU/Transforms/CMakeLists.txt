add_mlir_dialect_library(MLIRAMDGPUTransforms
  EmulateAtomics.cpp
  FoldMemRefsOps.cpp
  MaskedloadToLoad.cpp
  ResolveStridedMetadata.cpp
<<<<<<< HEAD
  MaskedloadToLoad.cpp
=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AMDGPU/Transforms

  DEPENDS
  MLIRAMDGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAMDGPUUtils
  MLIRAffineUtils
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  )
