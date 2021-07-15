add_mlir_dialect_library(MLIRTosaTransforms
  TosaMakeBroadcastable.cpp
  TosaPartition1.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa/Transforms

  DEPENDS
  MLIRTosaPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTosa
  )
