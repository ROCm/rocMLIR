set(LLVM_TARGET_DEFINITIONS MIGraphXToTosa.td)
mlir_tablegen(MIGraphXToTosa.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRMIGraphXToTosaConversionIncGen)

add_mlir_conversion_library(MLIRMIGraphXToTosa
  MIGraphXToTosa.cpp
  MIGraphXToTosaPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen
  MLIRMIGraphXToTosaConversionIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRTosaTransforms
  MLIRRockTransforms
  MLIRTransforms
  MLIRSupport
  MLIRMIGraphX
  )
