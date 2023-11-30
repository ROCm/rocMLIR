add_rocmlir_conversion_library(RocmlirEmulateFp8ExtTrunc
  EmulateFp8ExtTrunc.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef

  DEPENDS
  RocMLIRConversionPassIncGen
)

target_link_libraries(RocmlirEmulateFp8ExtTrunc
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRSupport
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRVectorDialect
)
