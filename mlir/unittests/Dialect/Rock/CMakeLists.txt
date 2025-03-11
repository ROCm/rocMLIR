add_rocmlir_unittest(MLIRRockUnitTests
  AmdArchDbTests.cpp
  TransformMapBuilderTests.cpp
)

target_link_libraries(MLIRRockUnitTests
  PRIVATE
  MLIRRockOps
  MLIRRockUtility
)
