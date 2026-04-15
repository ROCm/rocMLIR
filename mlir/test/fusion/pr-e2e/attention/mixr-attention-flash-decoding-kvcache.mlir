// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_attention_wrapper -relDiff_threshold 0.00001 -rand_min_int 0 -rand_max_int 4 -rand_type_int_for_inputs=2 --verifier clone - -pr | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CPU and GPU results may differ for masked-out positions. This is expected
// because flash decoding splits computation across splitKV blocks, each
// computing partial results that are combined using LSE (log-sum-exp)
// reduction. The LSE reduction takes place in a separate kernel that is
// provided by MIGraphX.

// CHECK: [0.583496,  -0.498047,  0.127441,  0.588867

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<2x6x1x2xf16, 12x2x2x1>,
                            %arg1: !migraphx.shaped<2x2x4x2xf16, 16x8x2x1>,
                            %arg2: !migraphx.shaped<2x1xsi32, 1x1>,
                            %arg3: !migraphx.shaped<2x2x4x2xf16, 16x8x2x1>) -> (!migraphx.shaped<2x2x1x4xf16, 8x4x4x1>, !migraphx.shaped<2x2x2x1x1xf32, 4x2x1x1x1>) attributes {rock.arch = "gfx950:sramecc+:xnack-", rock.kernel = "mixr", rock.num_cu = 256 : i64} {
    %0 = migraphx.literal(dense<[0, 1, 2, 3]> : tensor<4xsi32>) : <4xsi32, 1>
    %1 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.literal(dense<5.000000e-01> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.reshape %arg0 {dims = [2, 6, 1, 1, 2]} : <2x6x1x2xf16, 12x2x2x1> -> <2x6x1x1x2xf16, 12x2x2x2x1>
    %4 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [2, 6, 2, 1, 2]} : <2x6x1x1x2xf16, 12x2x2x2x1> -> <2x6x2x1x2xf16, 12x2x0x2x1>
    %5 = migraphx.reshape %arg1 {dims = [2, 2, 2, 2, 2]} : <2x2x4x2xf16, 16x8x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>
    %6 = migraphx.reshape %arg3 {dims = [2, 2, 2, 2, 2]} : <2x2x4x2xf16, 16x8x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>
    %7 = migraphx.slice %4 {axes = [1], ends = [2], starts = [0]} : <2x6x2x1x2xf16, 12x2x0x2x1> -> <2x2x2x1x2xf16, 12x2x0x2x1>
    %8 = migraphx.transpose %5 {permutation = [0, 1, 2, 4, 3]} : <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x2x2xf16, 16x8x4x1x2>
    %9 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [2, 2, 2, 1, 2]} : <1xf16, 1> -> <2x2x2x1x2xf16, 0x0x0x0x0>
    %10 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [2, 2, 2, 1, 2]} : <1xf16, 1> -> <2x2x2x1x2xf16, 0x0x0x0x0>
    %11 = migraphx.dot %7, %8 : <2x2x2x1x2xf16, 12x2x0x2x1>, <2x2x2x2x2xf16, 16x8x4x1x2> -> <2x2x2x1x2xf16, 8x4x2x2x1>
    %12 = migraphx.mul %11, %10 : <2x2x2x1x2xf16, 8x4x2x2x1>, <2x2x2x1x2xf16, 0x0x0x0x0> -> <2x2x2x1x2xf16, 8x4x2x2x1>
    %13 = migraphx.broadcast %0 {axis = 1 : i64, out_lens = [2, 4]} : <4xsi32, 1> -> <2x4xsi32, 0x1>
    %14 = migraphx.reshape %13 {dims = [2, 2, 2]} : <2x4xsi32, 0x1> -> <2x2x2xsi32, 4x2x1>
    %15 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [2, 4]} : <2x1xsi32, 1x1> -> <2x4xsi32, 1x0>
    %16 = migraphx.reshape %15 {dims = [2, 2, 2]} : <2x4xsi32, 1x0> -> <2x2x2xsi32, 4x2x1>
    %17 = migraphx.greater %14, %16 : <2x2x2xsi32, 4x2x1>, <2x2x2xsi32, 4x2x1> -> <2x2x2xsi32, 4x2x1>
    %18 = migraphx.convert %17 {target_type = 0 : i64} : <2x2x2xsi32, 4x2x1> to <2x2x2xsi8, 4x2x1>
    %19 = migraphx.reshape %18 {dims = [2, 1, 2, 1, 2]} : <2x2x2xsi8, 4x2x1> -> <2x1x2x1x2xsi8, 4x4x2x2x1>
    %20 = migraphx.multibroadcast %19 {out_dyn_dims = [], out_lens = [2, 2, 2, 1, 2]} : <2x1x2x1x2xsi8, 4x4x2x2x1> -> <2x2x2x1x2xsi8, 4x0x2x2x1>
    %21 = migraphx.where %20, %9, %12 : <2x2x2x1x2xsi8, 4x0x2x2x1>, <2x2x2x1x2xf16, 0x0x0x0x0>, <2x2x2x1x2xf16, 8x4x2x2x1> -> <2x2x2x1x2xf16, 8x4x2x2x1>
    %22 = migraphx.convert %21 {target_type = 2 : i64} : <2x2x2x1x2xf16, 8x4x2x2x1> to <2x2x2x1x2xf32, 8x4x2x2x1>
    %23 = migraphx.reshape %22 {dims = [2, 2, 2, 1, 2]} : <2x2x2x1x2xf32, 8x4x2x2x1> -> <2x2x2x1x2xf32, 8x4x2x2x1>
    %24 = migraphx.reduce_max %23 {axes = [4]} : <2x2x2x1x2xf32, 8x4x2x2x1> -> <2x2x2x1x1xf32, 4x2x1x1x1>
    %25 = migraphx.reshape %24 {dims = [2, 2, 2, 1, 1]} : <2x2x2x1x1xf32, 4x2x1x1x1> -> <2x2x2x1x1xf32, 4x2x1x1x1>
    %26 = migraphx.multibroadcast %25 {out_dyn_dims = [], out_lens = [2, 2, 2, 1, 2]} : <2x2x2x1x1xf32, 4x2x1x1x1> -> <2x2x2x1x2xf32, 4x2x1x1x0>
    %27 = migraphx.sub %22, %26 : <2x2x2x1x2xf32, 8x4x2x2x1>, <2x2x2x1x2xf32, 4x2x1x1x0> -> <2x2x2x1x2xf32, 8x4x2x2x1>
    %28 = migraphx.exp %27 : <2x2x2x1x2xf32, 8x4x2x2x1> -> <2x2x2x1x2xf32, 8x4x2x2x1>
    %29 = migraphx.reshape %28 {dims = [2, 2, 2, 1, 2]} : <2x2x2x1x2xf32, 8x4x2x2x1> -> <2x2x2x1x2xf32, 8x4x2x2x1>
    %30 = migraphx.reduce_sum %29 {axes = [4]} : <2x2x2x1x2xf32, 8x4x2x2x1> -> <2x2x2x1x1xf32, 4x2x1x1x1>
    %31 = migraphx.reshape %30 {dims = [2, 2, 2, 1, 1]} : <2x2x2x1x1xf32, 4x2x1x1x1> -> <2x2x2x1x1xf32, 4x2x1x1x1>
    %32 = migraphx.multibroadcast %31 {out_dyn_dims = [], out_lens = [2, 2, 2, 1, 2]} : <2x2x2x1x1xf32, 4x2x1x1x1> -> <2x2x2x1x2xf32, 4x2x1x1x0>
    %33 = migraphx.div %28, %32 : <2x2x2x1x2xf32, 8x4x2x2x1>, <2x2x2x1x2xf32, 4x2x1x1x0> -> <2x2x2x1x2xf32, 8x4x2x2x1>
    %34 = migraphx.convert %33 {target_type = 1 : i64} : <2x2x2x1x2xf32, 8x4x2x2x1> to <2x2x2x1x2xf16, 8x4x2x2x1>
    %35 = migraphx.dot %34, %6 : <2x2x2x1x2xf16, 8x4x2x2x1>, <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x1x2xf16, 8x4x2x2x1>
    %36 = migraphx.transpose %35 {permutation = [0, 2, 3, 1, 4]} : <2x2x2x1x2xf16, 8x4x2x2x1> -> <2x2x1x2x2xf16, 8x2x2x4x1>
    %37 = migraphx.reshape %36 {dims = [2, 2, 1, 4]} : <2x2x1x2x2xf16, 8x2x2x4x1> -> <2x2x1x4xf16, 8x4x4x1>
    %38 = migraphx.log %31 : <2x2x2x1x1xf32, 4x2x1x1x1> -> <2x2x2x1x1xf32, 4x2x1x1x1>
    %39 = migraphx.add %25, %38 : <2x2x2x1x1xf32, 4x2x1x1x1>, <2x2x2x1x1xf32, 4x2x1x1x1> -> <2x2x2x1x1xf32, 4x2x1x1x1>
    return %37, %39 : !migraphx.shaped<2x2x1x4xf16, 8x4x4x1>, !migraphx.shaped<2x2x2x1x1xf32, 4x2x1x1x1>
  }
}

