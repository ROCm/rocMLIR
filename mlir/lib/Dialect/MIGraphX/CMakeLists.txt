file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRMIGraphX
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIGraphX

  DEPENDS
  MLIRMIGraphXTypeIncGen
  MLIRMIGraphXOpsIncGen
  MLIRMIGraphXPassIncGen

  )
target_link_libraries(MLIRMIGraphX
  PUBLIC
  MLIRIR
  LLVMSupport
  MLIRPass
  MLIRFuncDialect
  MLIRTransforms
  MLIRSupport
)

add_subdirectory(Pipeline)
