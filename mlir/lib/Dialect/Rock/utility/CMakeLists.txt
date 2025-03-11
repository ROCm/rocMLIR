add_rocmlir_dialect_library(MLIRRockUtility
  AccelEmitter.cpp
  builderUtils.cpp
  loweringUtils.cpp
  transformMapUtils.cpp
  fusionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRRockOps
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRRockAnalysis
  MLIRMHAL
)

find_package(hip REQUIRED)
# We do need both of these.  One is for linking, to get the HIP
# functions.  The other is for compiling, to get -D__HIP_PLATFORM_AMD__
target_link_libraries(MLIRRockUtility PUBLIC
  hip::host hip::amdhip64
)
target_link_libraries(obj.MLIRRockUtility PUBLIC
  hip::host hip::amdhip64
)