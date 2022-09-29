// RUN: rocmlir-driver -host-pipeline highlevel %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float - | rocmlir-driver -c  | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
// CHECK: Unranked Memref base
  func.func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>) -> tensor<128x30x30x128xf32> attributes {kernel, arch = ""} {
  
    %zero = arith.constant dense<0.0> : tensor<128xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {
      dilation = [1, 1], 
      pad = [0, 0, 0, 0], 
      stride = [1, 1]
    }
     : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>) -> tensor<128x30x30x128xf32>

    %1 = "tosa.rsqrt"(%0)
     : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>
 
    return %1 : tensor<128x30x30x128xf32>
  }

} 

