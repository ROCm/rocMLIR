add_rocmlir_conversion_library(RocmlirCustomTosaDecompose
  RocmlirCustomTosaDecompose.cpp

  DEPENDS
  RocMLIRConversionPassIncGen
)

target_link_libraries(RocmlirCustomTosaDecompose
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRSupport
  MLIRArithDialect
  MLIRTosaDialect
  MLIRTensorDialect
  MLIRLinalgDialect
)
