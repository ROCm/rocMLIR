// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.0005  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x12x256x256xf32, 786432x65536x256x1>, %arg1: !migraphx.shaped<1x12x256x256xf32, 786432x65536x256x1>, %arg2: !migraphx.shaped<1x12x256x256xf32, 786432x65536x256x1>) -> (!migraphx.shaped<1x12x2x256x256xf32, 1572864x131072x65536x256x1>, !migraphx.shaped<1x12x2x256x1xf32, 6144x512x256x1x1>) attributes {kernel = "mixr"} {
    %0 = migraphx.reshape %arg0 {dims = [1, 12, 1, 256, 256]} : <1x12x256x256xf32, 786432x65536x256x1> -> <1x12x1x256x256xf32, 786432x65536x65536x256x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 256]} : <1x12x1x256x256xf32, 786432x65536x65536x256x1> -> <1x12x2x256x256xf32, 786432x65536x0x256x1>
    %2 = migraphx.reshape %arg1 {dims = [1, 12, 2, 128, 256]} : <1x12x256x256xf32, 786432x65536x256x1> -> <1x12x2x128x256xf32, 786432x65536x32768x256x1>
    %3 = migraphx.transpose %2 {permutation = [0, 1, 2, 4, 3]} : <1x12x2x128x256xf32, 786432x65536x32768x256x1> -> <1x12x2x256x128xf32, 786432x65536x32768x1x256>
    %4 = migraphx.reshape %arg2 {dims = [1, 12, 256, 2, 128]} : <1x12x256x256xf32, 786432x65536x256x1> -> <1x12x256x2x128xf32, 786432x65536x256x128x1>
    %5 = migraphx.transpose %4 {permutation = [0, 1, 3, 4, 2]} : <1x12x256x2x128xf32, 786432x65536x256x128x1> -> <1x12x2x128x256xf32, 786432x65536x128x1x256>
    %6 = migraphx.dot %1, %3 : <1x12x2x256x256xf32, 786432x65536x0x256x1>, <1x12x2x256x128xf32, 786432x65536x32768x1x256> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %7 = migraphx.reshape %6 {dims = [1, 12, 2, 256, 128]} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %8 = migraphx.reduce_max %7 {axes = [4]} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %9 = migraphx.reshape %8 {dims = [1, 12, 2, 256, 1]} : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %10 = migraphx.multibroadcast %9 {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 128]} : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x128xf32, 6144x512x256x1x0>
    %11 = migraphx.sub %6, %10 : <1x12x2x256x128xf32, 786432x65536x32768x128x1>, <1x12x2x256x128xf32, 6144x512x256x1x0> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %12 = migraphx.exp %11 : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %13 = migraphx.reshape %12 {dims = [1, 12, 2, 256, 128]} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %14 = migraphx.reduce_sum %13 {axes = [4]} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %15 = migraphx.reshape %14 {dims = [1, 12, 2, 256, 1]} : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %16 = migraphx.multibroadcast %15 {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 128]} : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x128xf32, 6144x512x256x1x0>
    %17 = migraphx.div %12, %16 : <1x12x2x256x128xf32, 786432x65536x32768x128x1>, <1x12x2x256x128xf32, 6144x512x256x1x0> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %18 = migraphx.dot %17, %5 : <1x12x2x256x128xf32, 786432x65536x32768x128x1>, <1x12x2x128x256xf32, 786432x65536x128x1x256> -> <1x12x2x256x256xf32, 1572864x131072x65536x256x1>
    %19 = migraphx.log %15 : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %20 = migraphx.add %9, %19 : <1x12x2x256x1xf32, 6144x512x256x1x1>, <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    return %18, %20 : !migraphx.shaped<1x12x2x256x256xf32, 1572864x131072x65536x256x1>, !migraphx.shaped<1x12x2x256x1xf32, 6144x512x256x1x1>
  }
}

