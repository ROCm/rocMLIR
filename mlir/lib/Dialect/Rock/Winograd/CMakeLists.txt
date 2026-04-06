set(WINOGRAD_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")

add_rocmlir_dialect_library(MLIRRockWinograd
  WinogradSolver.cpp
  WinogradArgLayout.cpp
  WinogradAssembler.cpp
  WinogradInterceptPass.cpp

  DEPENDS
  MLIRRockOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRRockOps
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRROCDLDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRFuncDialect
)

target_compile_definitions(MLIRRockWinograd PRIVATE
  ROCMLIR_WINOGRAD_KERNEL_DIR="${WINOGRAD_KERNEL_DIR}"
)
