// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-driver -host-pipeline=partition,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add --verifier clone - | rocmlir-driver -c -arch %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
// Note the fake wrapper function to make it look like this has already been partitioned,
// since we want "partition" for the xmodel packaging but don't actually want to use
// the partition logic.
// This is an awkward hack that should be fixed Later (tm)
module {
  func.func @mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add_real(%arg0: tensor<1x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32x64xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>, %arg5: tensor<32x64xf32>, %arg6: tensor<32x64xf32>, %arg7: tensor<128x64xf32>) -> tensor<32x64xf32> attributes {kernel = "mixr", num_cu = 48 : i64} {
    %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [32, 64]} : (tensor<1x64xf32>) -> tensor<32x64xf32>
    %1 = migraphx.transpose(%arg7) {permutation = [1, 0]} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %2 = migraphx.slice(%1) {axes = [1], ends = [128], starts = [64]} : (tensor<64x128xf32>) -> tensor<64x64xf32>
    %3 = migraphx.dot(%arg6, %2) : (tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<32x64xf32>
    %4 = migraphx.add(%3, %0) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %5 = migraphx.add(%arg1, %4) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %6 = migraphx.tanh(%5) : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %7 = migraphx.add(%arg2, %arg3) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %8 = migraphx.sigmoid(%7) : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %9 = migraphx.sub(%arg4, %8) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %10 = migraphx.mul(%9, %6) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %11 = migraphx.mul(%8, %arg5) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %12 = migraphx.add(%10, %11) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %12 : tensor<32x64xf32>
  }
  func.func @mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add(%arg0: tensor<1x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32x64xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>, %arg5: tensor<32x64xf32>, %arg6: tensor<32x64xf32>, %arg7: tensor<128x64xf32>) -> tensor<32x64xf32> {
    %ret = call @mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add_real(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<1x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<128x64xf32>) -> (tensor<32x64xf32>)
    return %ret : tensor<32x64xf32>
  }
}
