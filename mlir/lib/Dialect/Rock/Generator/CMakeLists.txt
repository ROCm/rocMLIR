set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(rocm_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)

add_rocmlir_dialect_library(MLIRRockConv2dGenerator
  AmdArchDb.cpp
  Conv2dGenerator.cpp

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
