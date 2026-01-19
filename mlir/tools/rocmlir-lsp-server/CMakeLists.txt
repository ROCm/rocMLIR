set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  MLIRLspServerLib
  # MLIRMHAL  # TODO: Re-enable when MHAL updated for LLVM 22
  # MLIRMHALTransforms
  MLIRAnalysis

  TritonIR
  TritonGPUIR
  TritonAnalysis
  TritonGPUTransforms
  TritonAMDGPUToLLVM
  TritonAMDGPUTransforms
  TritonAMDAnalysis
  TritonAMDGPUDialectToLLVM
  TritonAMDUtils
  GluonTransforms
  TritonInstrumentToLLVM
  NVHopperTransforms
  ProtonToProtonGPU
  ProtonGPUToLLVM
  ProtonNVIDIAGPUToLLVM
  ProtonAMDGPUToLLVM
  ProtonGPUTransforms
  TritonLLVMIR
  NVGPUToLLVM
  TritonInstrumentTransforms
  )

add_rocmlir_tool(rocmlir-lsp-server
  rocmlir-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(rocmlir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(rocmlir-lsp-server)

# Add Triton include directories for compilation
target_include_directories(rocmlir-lsp-server PRIVATE
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_PROJECT_DIR}/third_party/amd
  ${TRITON_BINARY_DIR}/third_party/amd
  ${TRITON_PROJECT_DIR}/third_party/nvidia
  ${TRITON_BINARY_DIR}/third_party/nvidia
  ${TRITON_PROJECT_DIR}/third_party
  ${TRITON_PROJECT_DIR}
  ${TRITON_BINARY_DIR}/third_party
  ${TRITON_BINARY_DIR}/third_party/amd/include
  ${TRITON_BINARY_DIR}/third_party/nvidia/include
)

mlir_check_all_link_libraries(rocmlir-lsp-server)
