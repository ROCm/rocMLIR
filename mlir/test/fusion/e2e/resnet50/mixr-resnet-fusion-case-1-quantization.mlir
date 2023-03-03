// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -verifier clone -fut test - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

module {
  func.func @test(%arg0: tensor<1x128x1x1xf32>, %arg1: tensor<1x128x56x56xi8>, %arg2: tensor<128x128x3x3xi8>) -> tensor<1x128x28x28xi8> {
    %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x128x56x56xi8>, tensor<128x128x3x3xi8>) -> tensor<1x128x28x28xi32>
    %2 = "migraphx.quantizelinear"(%1, %arg0) : (tensor<1x128x28x28xi32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xi8>
    return %2 : tensor<1x128x28x28xi8>
  }
}

//module {
//  func.func @test(%arg0: tensor<1x128x1x1xi32>, %arg1: tensor<1x128x56x56xi8>, %arg2: tensor<128x128x3x3xi8>) -> tensor<1x128x28x28xi32> {
//    %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x128x56x56xi8>, tensor<128x128x3x3xi8>) -> tensor<1x128x28x28xi32>
//    %2 = migraphx.add(%1, %arg0) : (tensor<1x128x28x28xi32>, tensor<1x128x1x1xi32>) -> tensor<1x128x28x28xi32>
//    return %2 : tensor<1x128x28x28xi32>
//  }
//}

//module {
//  func.func @test(%arg0: tensor<1x128x1x1xf32>, %arg1: tensor<1x128x56x56xi8>, %arg2: tensor<128x128x3x3xi8>) -> tensor<1x128x28x28xi32> {
//    %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x128x56x56xi8>, tensor<128x128x3x3xi8>) -> tensor<1x128x28x28xi32>
//    return %1 : tensor<1x128x28x28xi32>
//  }
//}
