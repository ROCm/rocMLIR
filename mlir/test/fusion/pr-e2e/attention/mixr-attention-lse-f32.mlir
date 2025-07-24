// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper --RMS_threshold 0.0003 -relDiff_threshold 0.0003  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]

module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg1: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg2: !migraphx.shaped<12x256x256xsi8, 65536x256x1>, %arg3: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg4: !migraphx.shaped<12x256x256xf16, 65536x256x1>) -> (!migraphx.shaped<12x256x256xf16, 65536x256x1>, !migraphx.shaped<12x256x1xf32, 256x1x1>) {
    %0 = migraphx.literal(dense<1.000000e+01> : tensor<12x256x256xf16>) : <12x256x256xf16, 65536x256x1>
    %1 = migraphx.literal(dense<1.250000e-01> : tensor<12x256x256xf16>) : <12x256x256xf16, 65536x256x1>
    %2 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x1x256>
    %3 = migraphx.transpose %arg3 {permutation = [0, 1, 3, 2]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x1x256>
    %4 = migraphx.reshape %3 {dims = [12, 256, 256]} : <1x12x256x256xf16, 786432x65536x1x256> -> <12x256x256xf16, 65536x1x256>
    %5 = migraphx.dot %arg0, %2 : <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 786432x65536x1x256> -> <1x12x256x256xf16, 786432x65536x256x1>
    %6 = migraphx.reshape %5 {dims = [12, 256, 256]} : <1x12x256x256xf16, 786432x65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %7 = migraphx.mul %6, %1 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %8 = migraphx.where %arg2, %7, %0 : <12x256x256xsi8, 65536x256x1>, <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %10 = migraphx.reshape %8 {dims = [12, 256, 256]} : <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %11 = migraphx.reduce_max %10 {axes = [2]} : <12x256x256xf16, 65536x256x1> -> <12x256x1xf16, 256x1x1>
    %12 = migraphx.reshape %11 {dims = [12, 256, 1]} : <12x256x1xf16, 256x1x1> -> <12x256x1xf16, 256x1x1>
    %13 = migraphx.multibroadcast %12 {out_dyn_dims = [], out_lens = [12, 256, 256]} : <12x256x1xf16, 256x1x1> -> <12x256x256xf16, 256x1x0>
    %14 = migraphx.sub %8, %13 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 256x1x0> -> <12x256x256xf16, 65536x256x1>
    %15 = migraphx.exp %14 : <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %16 = migraphx.reshape %15 {dims = [12, 256, 256]} : <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %17 = migraphx.reduce_sum %16 {axes = [2]} : <12x256x256xf16, 65536x256x1> -> <12x256x1xf16, 256x1x1>
    %18 = migraphx.reshape %17 {dims = [12, 256, 1]} : <12x256x1xf16, 256x1x1> -> <12x256x1xf16, 256x1x1>
    %19 = migraphx.multibroadcast %18 {out_dyn_dims = [], out_lens = [12, 256, 256]} : <12x256x1xf16, 256x1x1> -> <12x256x256xf16, 256x1x0>
    %20 = migraphx.div %15, %19 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 256x1x0> -> <12x256x256xf16, 65536x256x1>
    
    %se = migraphx.convert %17 {target_type = 2 : i64} : <12x256x1xf16, 256x1x1> to <12x256x1xf32, 256x1x1>
    %max = migraphx.convert %11 {target_type = 2 : i64} : <12x256x1xf16, 256x1x1> to <12x256x1xf32, 256x1x1>
    %lse = migraphx.log %se : <12x256x1xf32, 256x1x1> -> <12x256x1xf32, 256x1x1>
    %lse_add = migraphx.add %lse, %max : <12x256x1xf32, 256x1x1>, <12x256x1xf32, 256x1x1> -> <12x256x1xf32, 256x1x1>
    %22 = migraphx.dot %20, %4 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 65536x1x256> -> <12x256x256xf16, 65536x256x1>
    %23 = migraphx.mul %22, %arg4 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    return %23, %lse_add : !migraphx.shaped<12x256x256xf16, 65536x256x1>, !migraphx.shaped<12x256x1xf32, 256x1x1>
  }
}
