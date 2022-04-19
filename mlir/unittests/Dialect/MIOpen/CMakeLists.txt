add_mlir_miopen_unittest(MLIRMIOpenTransformMapBuilderTests
  TransformMapBuilderTests.cpp
)

target_link_libraries(MLIRMIOpenTransformMapBuilderTests
  PRIVATE
  MLIRMIOpenOps
)
