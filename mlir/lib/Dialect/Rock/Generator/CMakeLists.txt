set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIRDialect
  MLIRRockOps
  MLIRRockUtility
  MLIRParser
  MLIRPass
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRTransforms
  MLIRSupport
  MLIRIR
)

add_mlir_dialect_library(MLIRRockConv2dGenerator
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
