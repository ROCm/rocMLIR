add_mlir_rock_unittest(MLIRRockTransformMapBuilderTests
  TransformMapBuilderTests.cpp
)

target_link_libraries(MLIRRockTransformMapBuilderTests
  PRIVATE
  MLIRRockOps
)
