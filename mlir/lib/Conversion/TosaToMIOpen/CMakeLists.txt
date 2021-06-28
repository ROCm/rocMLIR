add_mlir_conversion_library(MLIRTosaToMIOpen
  TosaToMIOpen.cpp
  TosaToMIOpenPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRMIOpenTransforms
  MLIRLinalgUtils
  MLIRPass
  MLIRTosa
  MLIRTosaTransforms
  MLIRSupport
  )
