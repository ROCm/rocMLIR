set(LLVM_LINK_COMPONENTS
  Support
  )

add_rocmlir_dialect_library(MLIRRockConv2dGenerator
  ConvGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock/Generator

  DEPENDS
  MLIRSupport
)

target_link_libraries(MLIRRockConv2dGenerator
  PRIVATE
  MLIRRockTransforms
  MLIRRockOps
)
