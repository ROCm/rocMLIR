// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -RMS_threshold 0.01  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg1: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg2: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg3: !migraphx.shaped<1xsi32, 1>) -> (!migraphx.shaped<1x12x2x256x256xf16, 1572864x131072x65536x256x1>, !migraphx.shaped<1x12x2x256x1xf32, 6144x512x256x1x1>) attributes {kernel = "mixr", arch=""} {
    %mask_value = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %0 = migraphx.reshape %arg0 {dims = [1, 12, 1, 256, 256]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x1x256x256xf16, 786432x65536x65536x256x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 256]} : <1x12x1x256x256xf16, 786432x65536x65536x256x1> -> <1x12x2x256x256xf16, 786432x65536x0x256x1>
    %2 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x1x256>
    %3 = migraphx.reshape %2 {dims = [1, 12, 2, 256, 128]} : <1x12x256x256xf16, 786432x65536x1x256> -> <1x12x2x256x128xf16, 786432x65536x32768x128x1>
    %4 = migraphx.reshape %arg2 {dims = [1, 12, 256, 2, 128]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x2x128xf16, 786432x65536x256x128x1>
    %5 = migraphx.transpose %4 {permutation = [0, 1, 3, 4, 2]} : <1x12x256x2x128xf16, 786432x65536x256x128x1> -> <1x12x2x128x256xf16, 786432x65536x128x1x256>
    %6 = migraphx.dot %1, %3 : <1x12x2x256x256xf16, 786432x65536x0x256x1>, <1x12x2x256x128xf16, 786432x65536x32768x128x1> -> <1x12x2x256x128xf16, 786432x65536x32768x128x1>
    %7 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xsi32>) : <128xsi32, 1>
    %8 = migraphx.multibroadcast %7 {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 128]} : <128xsi32, 1> -> <1x12x2x256x128xsi32, 0x0x0x0x1>
    %9 = migraphx.reshape %arg3 {dims = [1, 1, 1, 1, 1]} : <1xsi32, 1> -> <1x1x1x1x1xsi32, 1x1x1x1x1>
    %10 = migraphx.multibroadcast %9 {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 128]} : <1x1x1x1x1xsi32, 1x1x1x1x1> -> <1x12x2x256x128xsi32, 0x0x0x0x0>
    %11 = migraphx.greater %8, %10 : <1x12x2x256x128xsi32, 0x0x0x0x1>, <1x12x2x256x128xsi32, 0x0x0x0x0> -> <1x12x2x256x128xsi32, 786432x65536x32768x128x1>
    %12 = migraphx.convert %11 {target_type = 0 : i64} : <1x12x2x256x128xsi32, 786432x65536x32768x128x1> to <1x12x2x256x128xsi8, 786432x65536x32768x128x1>
    %13 = migraphx.multibroadcast %mask_value {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 128]} : <1xf16, 1> -> <1x12x2x256x128xf16, 0x0x0x0x0>
    %14 = migraphx.where %12, %13, %6 : <1x12x2x256x128xsi8, 786432x65536x32768x128x1>, <1x12x2x256x128xf16, 0x0x0x0x0>, <1x12x2x256x128xf16, 786432x65536x32768x128x1> -> <1x12x2x256x128xf16, 786432x65536x32768x128x1>
    %15 = migraphx.convert %14 {target_type = 2 : i64} : <1x12x2x256x128xf16, 786432x65536x32768x128x1> to <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %16 = migraphx.reshape %15 {dims = [1, 12, 2, 256, 128]} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %17 = migraphx.reduce_max %16 {axes = [4]} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %18 = migraphx.reshape %17 {dims = [1, 12, 2, 256, 1]} : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %19 = migraphx.multibroadcast %18 {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 128]} : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x128xf32, 6144x512x256x1x0>
    %20 = migraphx.sub %15, %19 : <1x12x2x256x128xf32, 786432x65536x32768x128x1>, <1x12x2x256x128xf32, 6144x512x256x1x0> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %21 = migraphx.exp %20 : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %22 = migraphx.reshape %21 {dims = [1, 12, 2, 256, 128]} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %23 = migraphx.reduce_sum %22 {axes = [4]} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %24 = migraphx.reshape %23 {dims = [1, 12, 2, 256, 1]} : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %25 = migraphx.multibroadcast %24 {out_dyn_dims = [], out_lens = [1, 12, 2, 256, 128]} : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x128xf32, 6144x512x256x1x0>
    %26 = migraphx.div %21, %25 : <1x12x2x256x128xf32, 786432x65536x32768x128x1>, <1x12x2x256x128xf32, 6144x512x256x1x0> -> <1x12x2x256x128xf32, 786432x65536x32768x128x1>
    %27 = migraphx.convert %26 {target_type = 1 : i64} : <1x12x2x256x128xf32, 786432x65536x32768x128x1> to <1x12x2x256x128xf16, 786432x65536x32768x128x1>
    %28 = migraphx.dot %27, %5 : <1x12x2x256x128xf16, 786432x65536x32768x128x1>, <1x12x2x128x256xf16, 786432x65536x128x1x256> -> <1x12x2x256x256xf16, 1572864x131072x65536x256x1>
    %29 = migraphx.log %24 : <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    %30 = migraphx.add %18, %29 : <1x12x2x256x1xf32, 6144x512x256x1x1>, <1x12x2x256x1xf32, 6144x512x256x1x1> -> <1x12x2x256x1xf32, 6144x512x256x1x1>
    return %28, %30 : !migraphx.shaped<1x12x2x256x256xf16, 1572864x131072x65536x256x1>, !migraphx.shaped<1x12x2x256x1xf32, 6144x512x256x1x1>
  }
}

