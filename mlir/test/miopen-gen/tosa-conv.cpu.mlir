// RUN: miopen-opt -pass-pipeline="func.func(tosa-to-linalg-named)" -pass-pipeline="func.func(tosa-to-linalg)" -linalg-fuse-elementwise-ops -linalg-bufferize -func-bufferize -buffer-results-to-out-params -finalizing-bufferize -miopen-copy-opt  %s | miopen-gen -ph -pr -rand 1 -fut test_fusion - | miopen-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK:  [0,     0,     6,     6,     0,     6,     0,     0,     3,     6,     0,     6,     6,     6,     6,     6]
  func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x3x3x8xf32>) -> tensor<1x30x30x16xf32> {

    %cst = arith.constant dense<0.0> : tensor<16xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst) {
      dilation = [1, 1],
      pad = [0, 0, 0, 0],
      stride = [1, 1]
    }
     : (tensor<1x32x32x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<1x30x30x16xf32>

    %1 = "tosa.clamp"(%0) {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 6 : i64
    }
     : (tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>

    return %1 : tensor<1x30x30x16xf32>
  }

}

