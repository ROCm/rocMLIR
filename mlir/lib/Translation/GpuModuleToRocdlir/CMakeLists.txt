add_mlir_translation_library(GpuModuleToRocdlirTranslation
  GpuModuleToRocdlir.cpp

  DEPENDS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPUOps
  MLIRLLVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation

  MLIRIR
  MLIRLLVMDialect
  MLIRROCDLDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
)
