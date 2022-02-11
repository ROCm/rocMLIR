add_mlir_translation_library(GpuModuleToRocdlirTranslation
  GpuModuleToRocdlir.cpp

  DEPENDS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPUOps
  MLIRROCDLToLLVMIRTranslation

  MLIRIR
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRSupport
  MLIRTargetLLVMIRExport
)
