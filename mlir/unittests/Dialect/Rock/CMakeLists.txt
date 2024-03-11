add_rocmlir_unittest(MLIRRockTransformMapBuilderTests
  TransformMapBuilderTests.cpp
)

target_link_libraries(MLIRRockTransformMapBuilderTests
  PRIVATE
  MLIRRockOps
  MLIRRockUtility
)
