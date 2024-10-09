add_mlir_translation_library(GpuModuleToRocdlirTranslation
  GpuModuleToRocdlir.cpp

  DEPENDS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRGPUDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation

  MLIRIR
  MLIRLLVMDialect
  MLIRROCDLDialect
  MLIRPtrDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
)
