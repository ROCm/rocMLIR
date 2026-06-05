add_rocmlir_conversion_library(RocmlirFixTosaCastRounding
  FixTosaCastRounding.cpp

  DEPENDS
  RocMLIRConversionPassIncGen
)

target_link_libraries(RocmlirFixTosaCastRounding
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTransformUtils
)
