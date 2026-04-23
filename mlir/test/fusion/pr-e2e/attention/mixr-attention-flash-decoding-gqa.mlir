// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.001 -RMS_threshold 0.002 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x4x256x256xf16, 262144x65536x256x1>, %arg1: !migraphx.shaped<1x2x256x256xf16, 131072x65536x256x1>, %arg2: !migraphx.shaped<1x2x256x256xf16, 131072x65536x256x1>) -> (!migraphx.shaped<1x4x2x256x256xf16, 524288x131072x65536x256x1>, !migraphx.shaped<1x4x2x256x1xf32, 2048x512x256x1x1>) attributes {rock.kernel = "mixr", rock.arch="gfx942"} {
    %0 = migraphx.reshape %arg0 {dims = [1, 4, 1, 256, 256]} : <1x4x256x256xf16, 262144x65536x256x1> -> <1x4x1x256x256xf16, 262144x65536x65536x256x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 4, 2, 256, 256]} : <1x4x1x256x256xf16, 262144x65536x65536x256x1> -> <1x4x2x256x256xf16, 262144x65536x0x256x1>
    %2 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x2x256x256xf16, 131072x65536x256x1> -> <1x2x256x256xf16, 131072x65536x1x256>
    %3 = migraphx.reshape %2 {dims = [1, 2, 2, 256, 128]} : <1x2x256x256xf16, 131072x65536x1x256> -> <1x2x2x256x128xf16, 131072x65536x32768x128x1>
    %4 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [1, 2, 2, 2, 256, 128]} : <1x2x2x256x128xf16, 131072x65536x32768x128x1> -> <1x2x2x2x256x128xf16, 131072x65536x32768x0x128x1>
    %5 = migraphx.reshape %4 {dims = [1, 4, 2, 256, 128]} : <1x2x2x2x256x128xf16, 131072x65536x32768x0x128x1> -> <1x4x2x256x128xf16, 131072x65536x32768x128x1>
    %6 = migraphx.reshape %arg2 {dims = [1, 2, 256, 2, 128]} : <1x2x256x256xf16, 131072x65536x256x1> -> <1x2x256x2x128xf16, 131072x65536x256x128x1>
    %7 = migraphx.transpose %6 {permutation = [0, 1, 3, 4, 2]} : <1x2x256x2x128xf16, 131072x65536x256x128x1> -> <1x2x2x128x256xf16, 131072x65536x128x1x256>
    %8 = migraphx.multibroadcast %7 {out_dyn_dims = [], out_lens = [1, 2, 2, 2, 128, 256]} : <1x2x2x128x256xf16, 131072x65536x128x1x256> -> <1x2x2x2x128x256xf16, 131072x65536x128x0x1x256>
    %9 = migraphx.reshape %8 {dims = [1, 4, 2, 128, 256]} : <1x2x2x2x128x256xf16, 131072x65536x128x0x1x256> -> <1x4x2x128x256xf16, 131072x65536x128x1x256>
    %10 = migraphx.dot %1, %5 : <1x4x2x256x256xf16, 262144x65536x0x256x1>, <1x4x2x256x128xf16, 131072x65536x32768x128x1> -> <1x4x2x256x128xf16, 524288x131072x65536x128x1>
    %11 = migraphx.convert %10 {target_type = 2 : i64} : <1x4x2x256x128xf16, 524288x131072x65536x128x1> to <1x4x2x256x128xf32, 524288x131072x65536x128x1>
    %12 = migraphx.reshape %11 {dims = [1, 4, 2, 256, 128]} : <1x4x2x256x128xf32, 524288x131072x65536x128x1> -> <1x4x2x256x128xf32, 524288x131072x65536x128x1>
    %13 = migraphx.reduce_max %12 {axes = [4]} : <1x4x2x256x128xf32, 524288x131072x65536x128x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %14 = migraphx.reshape %13 {dims = [1, 4, 2, 256, 1]} : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %15 = migraphx.multibroadcast %14 {out_dyn_dims = [], out_lens = [1, 4, 2, 256, 128]} : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x128xf32, 2048x512x256x1x0>
    %16 = migraphx.sub %11, %15 : <1x4x2x256x128xf32, 524288x131072x65536x128x1>, <1x4x2x256x128xf32, 2048x512x256x1x0> -> <1x4x2x256x128xf32, 524288x131072x65536x128x1>
    %17 = migraphx.exp %16 : <1x4x2x256x128xf32, 524288x131072x65536x128x1> -> <1x4x2x256x128xf32, 524288x131072x65536x128x1>
    %18 = migraphx.reshape %17 {dims = [1, 4, 2, 256, 128]} : <1x4x2x256x128xf32, 524288x131072x65536x128x1> -> <1x4x2x256x128xf32, 524288x131072x65536x128x1>
    %19 = migraphx.reduce_sum %18 {axes = [4]} : <1x4x2x256x128xf32, 524288x131072x65536x128x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %20 = migraphx.reshape %19 {dims = [1, 4, 2, 256, 1]} : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %21 = migraphx.multibroadcast %20 {out_dyn_dims = [], out_lens = [1, 4, 2, 256, 128]} : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x128xf32, 2048x512x256x1x0>
    %22 = migraphx.div %17, %21 : <1x4x2x256x128xf32, 524288x131072x65536x128x1>, <1x4x2x256x128xf32, 2048x512x256x1x0> -> <1x4x2x256x128xf32, 524288x131072x65536x128x1>
    %23 = migraphx.convert %22 {target_type = 1 : i64} : <1x4x2x256x128xf32, 524288x131072x65536x128x1> to <1x4x2x256x128xf16, 524288x131072x65536x128x1>
    %24 = migraphx.dot %23, %9 : <1x4x2x256x128xf16, 524288x131072x65536x128x1>, <1x4x2x128x256xf16, 131072x65536x128x1x256> -> <1x4x2x256x256xf16, 524288x131072x65536x256x1>
    %25 = migraphx.log %20 : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %26 = migraphx.add %14, %25 : <1x4x2x256x1xf32, 2048x512x256x1x1>, <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    return %24, %26 : !migraphx.shaped<1x4x2x256x256xf16, 524288x131072x65536x256x1>, !migraphx.shaped<1x4x2x256x1xf32, 2048x512x256x1x1>
  }
}


