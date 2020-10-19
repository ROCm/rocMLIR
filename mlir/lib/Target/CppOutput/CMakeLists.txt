add_mlir_translation_library(MLIRTargetMIOpenCppTranslation
  ConvertToMIOpenCPP.cpp
  gridwise_convolution_implicit_gemm_v4r4.cpp
  gridwise_convolution_implicit_gemm_v4r4_gen_xdlops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMIOpenOps
  MLIRMIOpenTuning
  MLIRStandardOps
  MLIRTranslation
  )

