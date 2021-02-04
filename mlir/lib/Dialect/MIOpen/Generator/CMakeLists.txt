add_mlir_dialect_library(MLIRMIOpenConv2dGenerator
  Conv2dGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen/Generator

  DEPENDS
  MLIRSupport
)

target_link_libraries(MLIRMIOpenConv2dGenerator
  PRIVATE
  MLIRMIOpenOps
)
