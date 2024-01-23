add_rocmlir_unittest(MLIRRockTransformMapBuilderTests
  CollapseContiguousMergesTest.cpp
  TransformMapBuilderTests.cpp
)

target_link_libraries(MLIRRockTransformMapBuilderTests
  PRIVATE
  MLIRRockOps
)
