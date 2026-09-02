// REQUIRES: rocm-runner
// RUN: rocmlir-driver --host-pipeline=runner %s \
// RUN: | mlir-runner -O2 --shared-libs=%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void \
// RUN: | FileCheck %s

// Identical all-zero tensors have zero absolute error and zero scale. Verify
// that normalized RMS treats this exact match as zero instead of computing 0/0.

// CHECK: [1 1 1]

module {
  func.func @main() {
    %gpu = memref.alloc() : memref<4xf32>
    %reference = memref.alloc() : memref<4xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%gpu : memref<4xf32>)
    linalg.fill ins(%zero : f32) outs(%reference : memref<4xf32>)

    %gpuDynamic = memref.cast %gpu : memref<4xf32> to memref<?xf32>
    %referenceDynamic = memref.cast %reference : memref<4xf32> to memref<?xf32>
    %threshold = arith.constant 0.0 : f32
    %printDebug = arith.constant 0 : i8
    %isFP32 = arith.constant true
    %useAbsDiffGate = arith.constant false
    call @mcpuVerifyFloat(
        %gpuDynamic, %referenceDynamic, %threshold, %threshold, %threshold,
        %printDebug, %isFP32, %useAbsDiffGate)
        : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1, i1) -> ()

    memref.dealloc %gpu : memref<4xf32>
    memref.dealloc %reference : memref<4xf32>
    return
  }

  func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32,
                                    f32, i8, i1, i1)
}
