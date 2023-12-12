add_rocmlir_dialect_library(MLIRMIGraphXDialect
  IR/MIGraphX.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIGraphX

  DEPENDS
  MLIRMIGraphXTypeIncGen
  MLIRMIGraphXIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

add_subdirectory(Transforms)
add_subdirectory(Pipeline)
