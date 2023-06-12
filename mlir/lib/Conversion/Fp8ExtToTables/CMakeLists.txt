add_rocmlir_conversion_library(RocmlirFp8ExtToTables
  Fp8ExtToTables.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef

  DEPENDS
  RocMLIRConversionPassIncGen
)

target_link_libraries(RocmlirFp8ExtToTables
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRSupport
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRVectorDialect
)
