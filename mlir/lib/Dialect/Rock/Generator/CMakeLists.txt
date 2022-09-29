set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${dialect_libs}
  MLIRDialect
  MLIRRockOps
  MLIRRockUtility
  MLIRParser
  MLIRTransforms
  MLIRSupport
  MLIRIR
)

add_mlir_dialect_library(MLIRRockConv2dGenerator
  AmdArchDb.cpp
  Conv2dGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock/Generator

  DEPENDS
  MLIRSupport
)

target_link_libraries(MLIRRockConv2dGenerator
  PRIVATE
  MLIRIR
  MLIRRockOps
  ${LIBS}
)
