// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x1xsi32, 1x1>,
                            %arg1: !migraphx.shaped<1x6x1x2xf16, 12x2x2x1>,
                            %arg2: !migraphx.shaped<1x2x8x2xf16, 32x16x2x1>,
                            %arg3: !migraphx.shaped<1x2x8x2xf16, 32x16x2x1>) -> !migraphx.shaped<1x1x4xf16, 4x4x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.literal(dense<-3> : tensor<1xsi32>) : <1xsi32, 1>
    %1 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xsi32>) : <8xsi32, 1>
    %2 = migraphx.literal(dense<4> : tensor<1x1xsi32>) : <1x1xsi32, 1x1>
    %3 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xsi32>) : <8xsi32, 1>
    %4 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %5 = migraphx.literal(dense<5.000000e-01> : tensor<1xf16>) : <1xf16, 1>
    %6 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 1, 1, 8]} : <8xsi32, 1> -> <1x1x1x8xsi32, 0x0x0x1>
    %7 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 1, 1, 1]} : <1xsi32, 1> -> <1x1x1x1xsi32, 0x0x0x1>
    %8 = migraphx.reshape %arg0 {dims = [1, 1, 1, 1]} : <1x1xsi32, 1x1> -> <1x1x1x1xsi32, 1x1x1x1>
    %9 = migraphx.reshape %2 {dims = [1, 1, 1, 1]} : <1x1xsi32, 1x1> -> <1x1x1x1xsi32, 1x1x1x1>
    %10 = migraphx.reshape %2 {dims = [1, 1, 1, 1]} : <1x1xsi32, 1x1> -> <1x1x1x1xsi32, 1x1x1x1>
    %11 = migraphx.clip %8, %9, %10 : <1x1x1x1xsi32, 1x1x1x1>, <1x1x1x1xsi32, 1x1x1x1>, <1x1x1x1xsi32, 1x1x1x1> -> <1x1x1x1xsi32, 1x1x1x1>
    %12 = migraphx.slice %arg1 {axes = [1], ends = [2], starts = [0]} : <1x6x1x2xf16, 12x2x2x1> -> <1x2x1x2xf16, 12x2x2x1>
    %13 = migraphx.transpose %arg2 {permutation = [0, 1, 3, 2]} : <1x2x8x2xf16, 32x16x2x1> -> <1x2x2x8xf16, 32x16x1x2>
    %14 = migraphx.dot %12, %13 : <1x2x1x2xf16, 12x2x2x1>, <1x2x2x8xf16, 32x16x1x2> -> <1x2x1x8xf16, 16x8x8x1>
    %15 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [1, 2, 1, 8]} : <1xf16, 1> -> <1x2x1x8xf16, 0x0x0x0>
    %16 = migraphx.multibroadcast %5 {out_dyn_dims = [], out_lens = [1, 2, 1, 8]} : <1xf16, 1> -> <1x2x1x8xf16, 0x0x0x0>
    %17 = migraphx.mul %14, %16 : <1x2x1x8xf16, 16x8x8x1>, <1x2x1x8xf16, 0x0x0x0> -> <1x2x1x8xf16, 16x8x8x1>
    %18 = migraphx.add %11, %7 : <1x1x1x1xsi32, 1x1x1x1>, <1x1x1x1xsi32, 0x0x0x1> -> <1x1x1x1xsi32, 1x1x1x1>
    %19 = migraphx.multibroadcast %18 {out_dyn_dims = [], out_lens = [8, 1, 1, 1]} : <1x1x1x1xsi32, 1x1x1x1> -> <8x1x1x1xsi32, 0x1x1x1>
    %20 = migraphx.reshape %19 {dims = [8]} : <8x1x1x1xsi32, 0x1x1x1> -> <8xsi32, 0>
    %21 = migraphx.greater %20, %3 : <8xsi32, 0>, <8xsi32, 1> -> <8xsi32, 1>
    %22 = migraphx.convert %21 {target_type = 0 : i64} : <8xsi32, 1> to <8xsi8, 1>
    %23 = migraphx.broadcast %22 {axis = 3 : i64, out_lens = [1, 2, 1, 8]} : <8xsi8, 1> -> <1x2x1x8xsi8, 0x0x0x1>
    %24 = migraphx.where %23, %15, %17 : <1x2x1x8xsi8, 0x0x0x1>, <1x2x1x8xf16, 0x0x0x0>, <1x2x1x8xf16, 16x8x8x1> -> <1x2x1x8xf16, 16x8x8x1>
    %25 = migraphx.multibroadcast %11 {out_dyn_dims = [], out_lens = [1, 1, 1, 8]} : <1x1x1x1xsi32, 1x1x1x1> -> <1x1x1x8xsi32, 1x1x1x0>
    %26 = migraphx.greater %6, %25 : <1x1x1x8xsi32, 0x0x0x1>, <1x1x1x8xsi32, 1x1x1x0> -> <1x1x1x8xsi32, 0x0x0x1>
    %27 = migraphx.convert %26 {target_type = 0 : i64} : <1x1x1x8xsi32, 0x0x0x1> to <1x1x1x8xsi8, 0x0x0x1>
    %28 = migraphx.multibroadcast %27 {out_dyn_dims = [], out_lens = [1, 2, 1, 8]} : <1x1x1x8xsi8, 0x0x0x1> -> <1x2x1x8xsi8, 0x0x0x1>
    %29 = migraphx.where %28, %15, %24 : <1x2x1x8xsi8, 0x0x0x1>, <1x2x1x8xf16, 0x0x0x0>, <1x2x1x8xf16, 16x8x8x1> -> <1x2x1x8xf16, 16x8x8x1>
    %30 = migraphx.convert %29 {target_type = 2 : i64} : <1x2x1x8xf16, 16x8x8x1> to <1x2x1x8xf32, 16x8x8x1>
    %31 = migraphx.reshape %30 {dims = [1, 2, 1, 8]} : <1x2x1x8xf32, 16x8x8x1> -> <1x2x1x8xf32, 16x8x8x1>
    %32 = migraphx.reduce_max %31 {axes = [3]} : <1x2x1x8xf32, 16x8x8x1> -> <1x2x1x1xf32, 2x1x1x1>
    %33 = migraphx.reshape %32 {dims = [1, 2, 1, 1]} : <1x2x1x1xf32, 2x1x1x1> -> <1x2x1x1xf32, 2x1x1x1>
    %34 = migraphx.multibroadcast %33 {out_dyn_dims = [], out_lens = [1, 2, 1, 8]} : <1x2x1x1xf32, 2x1x1x1> -> <1x2x1x8xf32, 2x1x1x0>
    %35 = migraphx.sub %30, %34 : <1x2x1x8xf32, 16x8x8x1>, <1x2x1x8xf32, 2x1x1x0> -> <1x2x1x8xf32, 16x8x8x1>
    %36 = migraphx.exp %35 : <1x2x1x8xf32, 16x8x8x1> -> <1x2x1x8xf32, 16x8x8x1>
    %37 = migraphx.reshape %36 {dims = [1, 2, 1, 8]} : <1x2x1x8xf32, 16x8x8x1> -> <1x2x1x8xf32, 16x8x8x1>
    %38 = migraphx.reduce_sum %37 {axes = [3]} : <1x2x1x8xf32, 16x8x8x1> -> <1x2x1x1xf32, 2x1x1x1>
    %39 = migraphx.reshape %38 {dims = [1, 2, 1, 1]} : <1x2x1x1xf32, 2x1x1x1> -> <1x2x1x1xf32, 2x1x1x1>
    %40 = migraphx.multibroadcast %39 {out_dyn_dims = [], out_lens = [1, 2, 1, 8]} : <1x2x1x1xf32, 2x1x1x1> -> <1x2x1x8xf32, 2x1x1x0>
    %41 = migraphx.div %36, %40 : <1x2x1x8xf32, 16x8x8x1>, <1x2x1x8xf32, 2x1x1x0> -> <1x2x1x8xf32, 16x8x8x1>
    %42 = migraphx.convert %41 {target_type = 1 : i64} : <1x2x1x8xf32, 16x8x8x1> to <1x2x1x8xf16, 16x8x8x1>
    %43 = migraphx.dot %42, %arg3 : <1x2x1x8xf16, 16x8x8x1>, <1x2x8x2xf16, 32x16x2x1> -> <1x2x1x2xf16, 4x2x2x1>
    %44 = migraphx.transpose %43 {permutation = [0, 2, 1, 3]} : <1x2x1x2xf16, 4x2x2x1> -> <1x1x2x2xf16, 4x2x2x1>
    %45 = migraphx.reshape %44 {dims = [1, 1, 4]} : <1x1x2x2xf16, 4x2x2x1> -> <1x1x4xf16, 4x4x1>
    return %45 : !migraphx.shaped<1x1x4xf16, 4x4x1>
  }
}
