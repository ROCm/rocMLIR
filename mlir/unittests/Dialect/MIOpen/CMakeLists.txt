add_mlir_miopen_unittest(MLIRMIOpenCoordTransformBuilderTests
  CoordTransformBuilderTests.cpp
)

target_link_libraries(MLIRMIOpenCoordTransformBuilderTests
  PRIVATE
  MLIRMIOpenOps
)
