add_rocmlir_conversion_library(RocmlirTosaToLinalg
  RocmlirTosaToLinalg.cpp

  DEPENDS
  RocMLIRConversionPassIncGen
)

target_link_libraries(RocmlirTosaToLinalg
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
