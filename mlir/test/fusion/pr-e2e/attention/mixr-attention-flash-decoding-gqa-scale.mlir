// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.001  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x4x256x128xf16, 131072x32768x128x1>, %arg1: !migraphx.shaped<1x4x256x128xf16, 131072x32768x128x1>, %arg2: !migraphx.shaped<1x4x256x128xf16, 131072x32768x128x1>) -> (!migraphx.shaped<1x4x2x256x128xf16, 262144x65536x32768x128x1>, !migraphx.shaped<1x4x2x256x1xf32, 2048x512x256x1x1>) attributes {rock.kernel = "mixr", rock.arch="gfx942"} {
    %scale = migraphx.literal(dense<0.0883789> : tensor<1xf16>) : <1xf16, 1>
    %0 = migraphx.reshape %arg0 {dims = [1, 4, 1, 256, 128]} : <1x4x256x128xf16, 131072x32768x128x1> -> <1x4x1x256x128xf16, 131072x32768x32768x128x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 4, 2, 256, 128]} : <1x4x1x256x128xf16, 131072x32768x32768x128x1> -> <1x4x2x256x128xf16, 131072x32768x0x128x1>
    %2 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x4x256x128xf16, 131072x32768x128x1> -> <1x4x128x256xf16, 131072x32768x1x128>
    %3 = migraphx.reshape %2 {dims = [1, 4, 2, 128, 128]} : <1x4x128x256xf16, 131072x32768x1x128> -> <1x4x2x128x128xf16, 131072x32768x16384x128x1>
    %4 = migraphx.reshape %arg2 {dims = [1, 4, 128, 2, 128]} : <1x4x256x128xf16, 131072x32768x128x1> -> <1x4x128x2x128xf16, 131072x32768x256x128x1>
    %5 = migraphx.transpose %4 {permutation = [0, 1, 3, 4, 2]} : <1x4x128x2x128xf16, 131072x32768x256x128x1> -> <1x4x2x128x128xf16, 131072x32768x128x1x256>
    %6 = migraphx.dot %1, %3 : <1x4x2x256x128xf16, 131072x32768x0x128x1>, <1x4x2x128x128xf16, 131072x32768x16384x128x1> -> <1x4x2x256x128xf16, 262144x65536x32768x128x1>
    %7 = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [1, 4, 2, 256, 128]} : <1xf16, 1> -> <1x4x2x256x128xf16, 0x0x0x0x0>
    %8 = migraphx.mul %6, %7 : <1x4x2x256x128xf16, 262144x65536x32768x128x1>, <1x4x2x256x128xf16, 0x0x0x0x0> -> <1x4x2x256x128xf16, 262144x65536x32768x128x1>
    %9 = migraphx.convert %8 {target_type = 2 : i64} : <1x4x2x256x128xf16, 262144x65536x32768x128x1> to <1x4x2x256x128xf32, 262144x65536x32768x128x1>
    %10 = migraphx.reshape %9 {dims = [1, 4, 2, 256, 128]} : <1x4x2x256x128xf32, 262144x65536x32768x128x1> -> <1x4x2x256x128xf32, 262144x65536x32768x128x1>
    %11 = migraphx.reduce_max %10 {axes = [4]} : <1x4x2x256x128xf32, 262144x65536x32768x128x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %12 = migraphx.reshape %11 {dims = [1, 4, 2, 256, 1]} : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %13 = migraphx.multibroadcast %12 {out_dyn_dims = [], out_lens = [1, 4, 2, 256, 128]} : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x128xf32, 2048x512x256x1x0>
    %14 = migraphx.sub %9, %13 : <1x4x2x256x128xf32, 262144x65536x32768x128x1>, <1x4x2x256x128xf32, 2048x512x256x1x0> -> <1x4x2x256x128xf32, 262144x65536x32768x128x1>
    %15 = migraphx.exp %14 : <1x4x2x256x128xf32, 262144x65536x32768x128x1> -> <1x4x2x256x128xf32, 262144x65536x32768x128x1>
    %16 = migraphx.reshape %15 {dims = [1, 4, 2, 256, 128]} : <1x4x2x256x128xf32, 262144x65536x32768x128x1> -> <1x4x2x256x128xf32, 262144x65536x32768x128x1>
    %17 = migraphx.reduce_sum %16 {axes = [4]} : <1x4x2x256x128xf32, 262144x65536x32768x128x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %18 = migraphx.reshape %17 {dims = [1, 4, 2, 256, 1]} : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %19 = migraphx.multibroadcast %18 {out_dyn_dims = [], out_lens = [1, 4, 2, 256, 128]} : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x128xf32, 2048x512x256x1x0>
    %20 = migraphx.div %15, %19 : <1x4x2x256x128xf32, 262144x65536x32768x128x1>, <1x4x2x256x128xf32, 2048x512x256x1x0> -> <1x4x2x256x128xf32, 262144x65536x32768x128x1>
    %21 = migraphx.convert %20 {target_type = 1 : i64} : <1x4x2x256x128xf32, 262144x65536x32768x128x1> to <1x4x2x256x128xf16, 262144x65536x32768x128x1>
    %22 = migraphx.dot %21, %5 : <1x4x2x256x128xf16, 262144x65536x32768x128x1>, <1x4x2x128x128xf16, 131072x32768x128x1x256> -> <1x4x2x256x128xf16, 262144x65536x32768x128x1>
    %23 = migraphx.log %18 : <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    %24 = migraphx.add %12, %23 : <1x4x2x256x1xf32, 2048x512x256x1x1>, <1x4x2x256x1xf32, 2048x512x256x1x1> -> <1x4x2x256x1xf32, 2048x512x256x1x1>
    return %22, %24 : !migraphx.shaped<1x4x2x256x128xf16, 262144x65536x32768x128x1>, !migraphx.shaped<1x4x2x256x1xf32, 2048x512x256x1x1>
  }
}


