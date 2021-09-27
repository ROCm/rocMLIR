set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  LLVMROCmBackendUtils
  MLIRAnalysis
  MLIRDialect
  MLIRMIOpenOps
  MLIRParser
  MLIRPass
  MLIRQuant
  MLIRShape
  MLIRTransforms
  MLIRSupport
  MLIRIR
  MLIRTargetMIOpenCppTranslation
)

add_mlir_dialect_library(MLIRMIOpenConv2dGenerator
  Conv2dGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen/Generator

  DEPENDS
  MLIRSupport
)

target_link_libraries(MLIRMIOpenConv2dGenerator
  PRIVATE
  MLIRIR
  MLIRMIOpenOps
  ${LIBS}
)
