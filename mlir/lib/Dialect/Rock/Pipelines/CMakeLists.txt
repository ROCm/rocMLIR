add_rocmlir_dialect_library(MLIRRockPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRGPUToROCDLTransforms
  MLIRTensorToLinalg
  MLIRTosaToArith
  MLIRTosaToRock
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToTensor
  MLIRRockToGPU
  MLIRRockOps
  MLIRRockTransforms
  RocMLIRTransforms
)

if (ROCMLIR_ENABLE_COMGR)
  target_compile_definitions(obj.MLIRRockPipeline
  PRIVATE
  ROCMLIR_ENABLE_COMGR=1
  )
endif()
