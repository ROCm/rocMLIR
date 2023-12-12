// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-driver -host-pipeline=partition,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add --verifier clone - | rocmlir-driver -c -arch %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
// Note the fake wrapper function to make it look like this has already been partitioned,
// since we want "partition" for the xmodel packaging but don't actually want to use
// the partition logic.
// This is an awkward hack that should be fixed Later (tm)
module {
  func.func @mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add_real(%arg0: !migraphx.shaped<1x64xf32, 64x1>, %arg1: !migraphx.shaped<32x64xf32, 64x1>, %arg2: !migraphx.shaped<32x64xf32, 64x1>, %arg3: !migraphx.shaped<32x64xf32, 64x1>, %arg4: !migraphx.shaped<32x64xf32, 64x1>, %arg5: !migraphx.shaped<32x64xf32, 64x1>, %arg6: !migraphx.shaped<32x64xf32, 64x1>, %arg7: !migraphx.shaped<128x64xf32, 64x1>) -> !migraphx.shaped<32x64xf32, 64x1> attributes {kernel = "mixr", num_cu = 48 : i64} {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [32, 64]} : <1x64xf32, 64x1> -> <32x64xf32, 0x1>
    %1 = migraphx.transpose %arg7 {permutation = [1, 0]} : <128x64xf32, 64x1> -> <64x128xf32, 128x1>
    %2 = migraphx.slice %1 {axes = [1], ends = [128], starts = [64]} : <64x128xf32, 128x1> -> <64x64xf32, 64x1>
    %3 = migraphx.dot %arg6, %2 : <32x64xf32, 64x1>, <64x64xf32, 64x1> -> <32x64xf32, 64x1>
    %4 = migraphx.add %3, %0 : <32x64xf32, 64x1>, <32x64xf32, 0x1> -> <32x64xf32, 64x1>
    %5 = migraphx.add %arg1, %4 : <32x64xf32, 64x1>, <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    %6 = migraphx.tanh %5 : <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    %7 = migraphx.add %arg2, %arg3 : <32x64xf32, 64x1>, <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    %8 = migraphx.sigmoid %7 : <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    %9 = migraphx.sub %arg4, %8 : <32x64xf32, 64x1>, <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    %10 = migraphx.mul %9, %6 : <32x64xf32, 64x1>, <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    %11 = migraphx.mul %8, %arg5 : <32x64xf32, 64x1>, <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    %12 = migraphx.add %10, %11 : <32x64xf32, 64x1>, <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    return %12 : !migraphx.shaped<32x64xf32, 64x1>
  }
  func.func @mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add(%arg0: !migraphx.shaped<1x64xf32, 64x1>, %arg1: !migraphx.shaped<32x64xf32, 64x1>, %arg2: !migraphx.shaped<32x64xf32, 64x1>, %arg3: !migraphx.shaped<32x64xf32, 64x1>, %arg4: !migraphx.shaped<32x64xf32, 64x1>, %arg5: !migraphx.shaped<32x64xf32, 64x1>, %arg6: !migraphx.shaped<32x64xf32, 64x1>, %arg7: !migraphx.shaped<128x64xf32, 64x1>) -> !migraphx.shaped<32x64xf32, 64x1> {
    %ret = call @mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add_real(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (!migraphx.shaped<1x64xf32, 64x1>, !migraphx.shaped<32x64xf32, 64x1>, !migraphx.shaped<32x64xf32, 64x1>, !migraphx.shaped<32x64xf32, 64x1>, !migraphx.shaped<32x64xf32, 64x1>, !migraphx.shaped<32x64xf32, 64x1>, !migraphx.shaped<32x64xf32, 64x1>, !migraphx.shaped<128x64xf32, 64x1>) -> (!migraphx.shaped<32x64xf32, 64x1>)
    return %ret : !migraphx.shaped<32x64xf32, 64x1>
  }
}
