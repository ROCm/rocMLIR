// RUN: miopen-opt --tosa-to-miopen --tosa-to-linalg-on-tensors --linalg-fuse-elementwise-ops --linalg-bufferize --func-bufferize --buffer-results-to-out-params --finalizing-bufferize -miopen-copy-opt %s | miopen-gen -ph -print_results -rand 1 -rand_type float - | miopen-opt -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-linalg-align -convert-linalg-to-affine-loops -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
// CHECK: Unranked Memref base
  func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>) -> tensor<128x30x30x128xf32> {
  
    %zero = constant dense<0.0> : tensor<128xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {
      dilation = [1, 1], 
      pad = [0, 0, 0, 0], 
      stride = [1, 1]
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

