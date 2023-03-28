# Instead of using the LLVM and LLD living next to MLIR, use the COMGR interface
# to invoke the versions of these tools shipped with ROCm. This is used to
# avoid packaging a separate copy of the compiler when creating MLIR-based tools
# targetting ROCm, espcecially those living inside the ROCm distribution itself.
if (ROCMLIR_ENABLE_COMGR)
  find_package(amd_comgr REQUIRED PATHS /opt/rocm)
  set(COMGR_PASS_COMPONENTS
    Core
  )
  set(COMGR_PASS_LIBS)
endif()

add_rocmlir_dialect_library(RocMLIRTransforms
  CompileGpuWithComgr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/RocmlirTransforms

  LINK_COMPONENTS
  ${COMGR_PASS_COMPONENTS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRGPUOps
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRPass
  ${COMGR_PASS_LIBS}

  DEPENDS
  RocMLIRTransformsPassIncGen
)

if (ROCMLIR_ENABLE_COMGR)
  target_compile_definitions(obj.RocMLIRTransforms
  PRIVATE
  ROCMLIR_COMPILE_GPU_WITH_COMGR_PASS_ENABLE=1
  )

  target_link_libraries(RocMLIRTransforms PUBLIC amd_comgr)
endif()

