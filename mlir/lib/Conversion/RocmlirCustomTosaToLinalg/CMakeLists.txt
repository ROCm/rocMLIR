add_rocmlir_conversion_library(RocmlirCustomTosaToLinalg
  RocmlirCustomTosaToLinalg.cpp

  DEPENDS
  RocMLIRConversionPassIncGen
)

target_link_libraries(RocmlirCustomTosaToLinalg
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
