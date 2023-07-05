add_mlir_translation_library(GpuModuleToRocdlirTranslation
  GpuModuleToRocdlir.cpp

  DEPENDS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation

  MLIRIR
  MLIRLLVMDialect
  MLIRROCDLDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
)
