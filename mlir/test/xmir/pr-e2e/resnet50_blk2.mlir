// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch,gfx908,gfx90a %s | rocmlir-gen -ph -print-results -rand_type float -rand 1 -fut resnet50 - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full  -targets %arch,gfx908,gfx90a | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

module {
// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 32, 32, 64] strides = [65536, 2048, 64, 1] data =


  func.func @resnet50(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x3x3x64xf32>, %arg2: tensor<64x3x3x64xf32>, %arg3: tensor<64x3x3x64xf32>, %arg4: tensor<64x3x3x64xf32>) -> tensor<1x32x32x64xf32> {

    %cst = arith.constant dense<0.0> : tensor<64xf32>

    // Block 0
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
    %1 = "tosa.clamp"(%0) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    %2 = "tosa.conv2d"(%1, %arg2, %cst) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
    %3 = "tosa.clamp"(%2) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    %4 = "tosa.add"(%arg0, %3) : (tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    // Block 1
    %5 = "tosa.conv2d"(%4, %arg3, %cst) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
    %6 = "tosa.clamp"(%5) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    %7 = "tosa.conv2d"(%6, %arg4, %cst) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
    %8 = "tosa.clamp"(%7) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    %9 = "tosa.add"(%arg0, %8) : (tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    return %9 : tensor<1x32x32x64xf32>
  }
}

