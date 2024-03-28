if (MHAL_ENABLE_HOST_RUNNER)
  set(host_runner_libs
  MLIRAsyncTransforms
  MLIRGPUTransforms
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRGPUToROCDLTransforms
  MLIRReconcileUnrealizedCasts
  MLIRVectorToLLVMPass

  MLIRMHALToGPU
  MLIRMHALToCPU
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  )
else()
  set(host_runner_libs)
endif()

add_mlir_dialect_library(MLIRMHALPipeline
  Pipelines.cpp

  DEPENDS
  MHALConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRFuncDialect
  MLIRTensorToLinalg
  MLIRTosaToLinalg
  MLIRMHALTransforms
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils

  ${host_runner_libs}
)

if (MHAL_ENABLE_HOST_RUNNER)
  target_compile_definitions(obj.MLIRMHALPipeline
    PRIVATE
    MHAL_ENABLE_HOST_RUNNER=1)
endif()
