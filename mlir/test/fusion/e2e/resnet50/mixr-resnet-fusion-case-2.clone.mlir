// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 4 -rand_type float -fut func_rock --verifier clone - | rocmlir-driver -host-pipeline xmodel,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// XFAIL: *
module {
  // CHECK: Unranked Memref base
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  func.func @func_rock(%arg0: tensor<1x256x1x1xf32>, %arg1: tensor<1x256x56x56xf32>, %arg2: tensor<1x64x56x56xf32>, %arg3: tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32> attributes{kernel, arch = ""} {
    %0 = migraphx.multibroadcast(%arg0) {out_lens = [1, 256, 56, 56]} : (tensor<1x256x1x1xf32>) -> tensor<1x256x56x56xf32>
    %1 = migraphx.convolution(%arg2, %arg3) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %2 = migraphx.add(%1, %0) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %3 = migraphx.add(%2, %arg1) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %4 = migraphx.relu(%3) : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    return %4 : tensor<1x256x56x56xf32>
  }
}