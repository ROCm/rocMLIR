// This is a design that was in the MIGraphX CI that was previously failing
// here: https://ontrack-internal.amd.com/browse/SWDEV-558297

// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK: [1 1 1]
  func.func @mlir_attention(
        %arg0: !migraphx.shaped<1x1x12xf16, 12x12x1>,
        %arg1: !migraphx.shaped<1x2x4x2xf16, 16x8x2x1>,
        %arg2: !migraphx.shaped<1x2x4x2xf16, 16x8x2x1>,
        %arg3: !migraphx.shaped<1x1xsi32, 1x1>
    ) -> !migraphx.shaped<1x1x4xf16, 4x4x1> attributes {rock.arch = "", rock.kernel = "mixr", rock.num_cu = 0 : i64} {
    %0 = migraphx.literal(dense<[0, 1, 2, 3]> : tensor<4xsi32>) : <4xsi32, 1>
    %1 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.literal(dense<1.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.reshape %arg0 {dims = [1, 1, 6, 2]} : <1x1x12xf16, 12x12x1> -> <1x1x6x2xf16, 12x12x2x1>
    %4 = migraphx.transpose %3 {permutation = [0, 2, 1, 3]} : <1x1x6x2xf16, 12x12x2x1> -> <1x6x1x2xf16, 12x2x12x1>
    %5 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 2]} : <1x1xsi32, 1x1> -> <1x2xsi32, 1x0>
    %6 = migraphx.slice %4 {axes = [1], ends = [2], starts = [0]} : <1x6x1x2xf16, 12x2x12x1> -> <1x2x1x2xf16, 12x2x12x1>
    %7 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x2x4x2xf16, 16x8x2x1> -> <1x2x2x4xf16, 16x8x1x2>
    %8 = migraphx.dot %6, %7 : <1x2x1x2xf16, 12x2x12x1>, <1x2x2x4xf16, 16x8x1x2> -> <1x2x1x4xf16, 8x4x4x1>
    %9 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 2, 1, 4]} : <4xsi32, 1> -> <1x2x1x4xsi32, 0x0x0x1>
    %10 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 2, 1, 4]} : <1xf16, 1> -> <1x2x1x4xf16, 0x0x0x0>
    %11 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 2, 1, 4]} : <1xf16, 1> -> <1x2x1x4xf16, 0x0x0x0>
    %12 = migraphx.mul %8, %11 : <1x2x1x4xf16, 8x4x4x1>, <1x2x1x4xf16, 0x0x0x0> -> <1x2x1x4xf16, 8x4x4x1>
    %13 = migraphx.reshape %5 {dims = [1, 2, 1, 1]} : <1x2xsi32, 1x0> -> <1x2x1x1xsi32, 2x1x1x1>
    %14 = migraphx.multibroadcast %13 {out_dyn_dims = [], out_lens = [1, 2, 1, 4]} : <1x2x1x1xsi32, 2x1x1x1> -> <1x2x1x4xsi32, 2x1x1x0>
    %15 = migraphx.greater %9, %14 : <1x2x1x4xsi32, 0x0x0x1>, <1x2x1x4xsi32, 2x1x1x0> -> <1x2x1x4xsi32, 8x4x4x1>
    %16 = migraphx.convert %15 {target_type = 0 : i64} : <1x2x1x4xsi32, 8x4x4x1> to <1x2x1x4xsi8, 8x4x4x1>
    %17 = migraphx.where %16, %10, %12 : <1x2x1x4xsi8, 8x4x4x1>, <1x2x1x4xf16, 0x0x0x0>, <1x2x1x4xf16, 8x4x4x1> -> <1x2x1x4xf16, 8x4x4x1>
    %18 = migraphx.softmax %17 {axis = 3 : i64} : <1x2x1x4xf16, 8x4x4x1> -> <1x2x1x4xf16, 8x4x4x1>
    %19 = migraphx.dot %18, %arg2 : <1x2x1x4xf16, 8x4x4x1>, <1x2x4x2xf16, 16x8x2x1> -> <1x2x1x2xf16, 4x2x2x1>
    %20 = migraphx.transpose %19 {permutation = [0, 2, 1, 3]} : <1x2x1x2xf16, 4x2x2x1> -> <1x1x2x2xf16, 4x2x2x1>
    %21 = migraphx.reshape %20 {dims = [1, 1, 4]} : <1x1x2x2xf16, 4x2x2x1> -> <1x1x4xf16, 4x4x1>
    return %21 : !migraphx.shaped<1x1x4xf16, 4x4x1>
  }
}

