// RUN: mlir-miopen-driver -host-pipeline highlevel %s | miopen-gen -ph -print-results -rand 1 -rand_type float - | mlir-miopen-driver -c  | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
// CHECK: Unranked Memref base
  func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>) -> tensor<128x30x30x128xf32> attributes {kernel} {
  
    %zero = arith.constant dense<0.0> : tensor<128xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {
      dilation = [1, 1], 
      pad = [0, 0, 0, 0], 
      stride = [1, 1], xdlopsV2 = true
    }
     : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>) -> tensor<128x30x30x128xf32>

    %1 = "tosa.clamp"(%0) {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 6 : i64
    }
     : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>
 
    return %1 : tensor<128x30x30x128xf32>
  }

} 

