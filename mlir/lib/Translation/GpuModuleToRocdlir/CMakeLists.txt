add_mlir_translation_library(GpuModuleToRocdlirTranslation
  GpuModuleToRocdlir.cpp

  DEPENDS

  LINK_COMPONENTS
  Core
  Target
  MC
  Support
  SPIRV

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
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
