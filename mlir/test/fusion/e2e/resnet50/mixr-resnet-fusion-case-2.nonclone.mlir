  // RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
  
  // CHECK:  [1]
  func.func @func_ref(%arg0: tensor<1x256x1x1xf32>, %arg1: tensor<1x256x56x56xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32> {
    %0 = migraphx.multibroadcast(%arg0) {out_lens = [1, 256, 56, 56]} : (tensor<1x256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %1 = migraphx.convolution(%arg2, %arg3) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %2 = migraphx.add(%1, %0) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %3 = migraphx.add(%2, %arg1) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %4 = migraphx.relu(%3) : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    return %4 : tensor<1x256x56x56xf32>
  }

  func.func @func_rock(%arg0: tensor<1x256x1x1xf32>, %arg1: tensor<1x256x56x56xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32> attributes{kernel, arch = ""} {
    %0 = migraphx.multibroadcast(%arg0) {out_lens = [1, 256, 56, 56]} : (tensor<1x256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %1 = migraphx.convolution(%arg2, %arg3) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %2 = migraphx.add(%1, %0) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %3 = migraphx.add(%2, %arg1) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %4 = migraphx.relu(%3) : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    return %4 : tensor<1x256x56x56xf32>
  }

  func.func @test5(%arg0: tensor<1x256x1x1xf32>, %arg1: tensor<1x256x56x56xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<256x64x1x1xf32>) -> tensor<1x1x1x1xf32>{
    %ref = call @func_ref(%arg0, %arg1, %arg2, %arg3) : (tensor<1x256x1x1xf32>, tensor<1x256x56x56xf32>, tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> (tensor<1x256x56x56xf32>)
    %val = call @func_rock(%arg0, %arg1, %arg2, %arg3) : (tensor<1x256x1x1xf32>, tensor<1x256x56x56xf32>, tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> (tensor<1x256x56x56xf32>)
    %err = "tosa.sub"(%val, %ref) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %sum1 = "tosa.reduce_sum"(%err) {axis = 1 : i64} : (tensor<1x256x56x56xf32>) -> tensor<1x1x56x56xf32>
    %sum2 = "tosa.reduce_sum"(%sum1) {axis = 2 : i64} : (tensor<1x1x56x56xf32>) -> tensor<1x1x1x56xf32>
    %sum3 = "tosa.reduce_sum"(%sum2) {axis = 3 : i64} : (tensor<1x1x1x56xf32>) -> tensor<1x1x1x1xf32>
    return %sum3 : tensor<1x1x1x1xf32>
  }