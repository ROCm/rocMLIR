set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)
get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  ${rocmlir_test_libs}
  MLIROptLib
  MLIRAnalysis
  MLIRFuncInlinerExtension

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

if(MLIR_INCLUDE_TESTS)
  list(APPEND LIBS MLIRTestIR)
endif()

set(disable_install "")
if (BUILD_FAT_LIBROCKCOMPILER)
  set(disable_install "DISABLE_INSTALL")
endif()

add_mlir_library(RocMLIROptMain
  rocmlir-opt.cpp

  EXCLUDE_FROM_LIBMLIR
  ${disable_install}

  LINK_LIBS PRIVATE
  ${LIBS}
  )
set_target_properties(RocMLIROptMain PROPERTIES EXCLUDE_FROM_ALL ${BUILD_FAT_LIBROCKCOMPILER})

add_rocmlir_tool(rocmlir-opt
  rocmlir-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(rocmlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(rocmlir-opt)

# Add Triton include directories for compilation
target_include_directories(rocmlir-opt PRIVATE
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

mlir_check_all_link_libraries(rocmlir-opt)
