// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | FileCheck %s --check-prefix=FOLD
// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_attention_wrapper -relDiff_threshold 0.00001 -rand_min_int 1 -rand_max_int 2 -rand_type_int_for_inputs=2,4 --verifier clone - -pr | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

// Verify that all three nested selects are folded into one attention op.
// FOLD: rock.attention{
// FOLD-DAG: currentSeqLen = (
// FOLD-DAG: prefixOffset = (
// FOLD-DAG: slidingWindowSize = 1
// FOLD-DAG: causal
// FOLD: qk = elementwise {
// FOLD-NOT: tosa.select
// FOLD: rock.yield

// E2E: [1 1 1]
// E2E-NEXT: [1 1 1]

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<2x6x2x2xf16, 24x4x2x1>, %arg1: !migraphx.shaped<2x2x4x2xf16, 16x8x2x1>, %arg2: !migraphx.shaped<2x1xsi32, 1x1>, %arg3: !migraphx.shaped<2x2x4x2xf16, 16x8x2x1>, %arg4: !migraphx.shaped<2x1xsi32, 1x1>) -> (!migraphx.shaped<2x2x2x4xf16, 16x8x4x1>, !migraphx.shaped<2x2x2x2x1xf32, 8x4x2x1x1>) attributes {rock.kernel = "mixr"} {
    %0 = migraphx.literal(dense<[0, 1, 2, 3]> : tensor<4xsi32>) : <4xsi32, 1>
    %1 = migraphx.literal(dense<[[0], [1]]> : tensor<2x1xsi32>) : <2x1xsi32, 1x1>
    %2 = migraphx.literal(dense<1> : tensor<2x4xsi32>) : <2x4xsi32, 4x1>
    %3 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %4 = migraphx.literal(dense<5.000000e-01> : tensor<1xf16>) : <1xf16, 1>
    %sliding_offset = migraphx.literal(dense<-1> : tensor<1xsi32>) : <1xsi32, 1>
    // currentSeqLen is an inclusive key index, so the full-cache value is K-1.
    %fixed_seq_len = migraphx.literal(dense<1> : tensor<2x1xsi32>) : <2x1xsi32, 1x1>
    %seq_len = migraphx.clip %arg2, %fixed_seq_len, %fixed_seq_len : <2x1xsi32, 1x1>, <2x1xsi32, 1x1>, <2x1xsi32, 1x1> -> <2x1xsi32, 1x1>
    %5 = migraphx.reshape %arg0 {dims = [2, 6, 1, 2, 2]} : <2x6x2x2xf16, 24x4x2x1> -> <2x6x1x2x2xf16, 24x4x4x2x1>
    %6 = migraphx.multibroadcast %5 {out_dyn_dims = [], out_lens = [2, 6, 2, 2, 2]} : <2x6x1x2x2xf16, 24x4x4x2x1> -> <2x6x2x2x2xf16, 24x4x0x2x1>
    %7 = migraphx.reshape %arg1 {dims = [2, 2, 2, 2, 2]} : <2x2x4x2xf16, 16x8x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>
    %8 = migraphx.reshape %arg3 {dims = [2, 2, 2, 2, 2]} : <2x2x4x2xf16, 16x8x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>
    %9 = migraphx.slice %6 {axes = [1], ends = [2], starts = [0]} : <2x6x2x2x2xf16, 24x4x0x2x1> -> <2x2x2x2x2xf16, 24x4x0x2x1>
    %10 = migraphx.transpose %7 {permutation = [0, 1, 2, 4, 3]} : <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x2x2xf16, 16x8x4x1x2>
    %11 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 2]} : <1xf16, 1> -> <2x2x2x2x2xf16, 0x0x0x0x0>
    %12 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 2]} : <1xf16, 1> -> <2x2x2x2x2xf16, 0x0x0x0x0>
    %13 = migraphx.dot %9, %10 : <2x2x2x2x2xf16, 24x4x0x2x1>, <2x2x2x2x2xf16, 16x8x4x1x2> -> <2x2x2x2x2xf16, 16x8x4x2x1>
    %14 = migraphx.mul %13, %12 : <2x2x2x2x2xf16, 16x8x4x2x1>, <2x2x2x2x2xf16, 0x0x0x0x0> -> <2x2x2x2x2xf16, 16x8x4x2x1>

    // Prefix-causal mask.
    %15 = migraphx.multibroadcast %arg4 {out_dyn_dims = [], out_lens = [2, 1]} : <2x1xsi32, 1x1> -> <2x1xsi32, 1x0>
    %16 = migraphx.add %1, %15 : <2x1xsi32, 1x1>, <2x1xsi32, 1x0> -> <2x1xsi32, 1x1>
    %17 = migraphx.multibroadcast %16 {out_dyn_dims = [], out_lens = [2, 4]} : <2x1xsi32, 1x1> -> <2x4xsi32, 1x0>
    %18 = migraphx.mul %17, %2 : <2x4xsi32, 1x0>, <2x4xsi32, 4x1> -> <2x4xsi32, 4x1>
    %19 = migraphx.broadcast %0 {axis = 1 : i64, out_lens = [2, 4]} : <4xsi32, 1> -> <2x4xsi32, 0x1>
    %20 = migraphx.mul %19, %2 : <2x4xsi32, 0x1>, <2x4xsi32, 4x1> -> <2x4xsi32, 4x1>
    %21 = migraphx.greater %20, %18 : <2x4xsi32, 4x1>, <2x4xsi32, 4x1> -> <2x4xsi32, 4x1>
    %22 = migraphx.convert %21 {target_type = 0 : i64} : <2x4xsi32, 4x1> to <2x4xsi8, 4x1>
    %23 = migraphx.reshape %22 {dims = [1, 1, 2, 2, 2]} : <2x4xsi8, 4x1> -> <1x1x2x2x2xsi8, 8x8x4x2x1>
    %24 = migraphx.multibroadcast %23 {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 2]} : <1x1x2x2x2xsi8, 8x8x4x2x1> -> <2x2x2x2x2xsi8, 0x0x4x2x1>
    %25 = migraphx.where %24, %11, %14 : <2x2x2x2x2xsi8, 0x0x4x2x1>, <2x2x2x2x2xf16, 0x0x0x0x0>, <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>

    // KV-cache mask.
    %26 = migraphx.broadcast %0 {axis = 1 : i64, out_lens = [2, 4]} : <4xsi32, 1> -> <2x4xsi32, 0x1>
    %27 = migraphx.multibroadcast %seq_len {out_dyn_dims = [], out_lens = [2, 4]} : <2x1xsi32, 1x1> -> <2x4xsi32, 1x0>
    %28 = migraphx.greater %26, %27 : <2x4xsi32, 0x1>, <2x4xsi32, 1x0> -> <2x4xsi32, 4x1>
    %29 = migraphx.convert %28 {target_type = 0 : i64} : <2x4xsi32, 4x1> to <2x4xsi8, 4x1>
    %30 = migraphx.reshape %29 {dims = [2, 1, 2, 1, 2]} : <2x4xsi8, 4x1> -> <2x1x2x1x2xsi8, 4x4x2x2x1>
    %31 = migraphx.multibroadcast %30 {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 2]} : <2x1x2x1x2xsi8, 4x4x2x2x1> -> <2x2x2x2x2xsi8, 4x0x2x0x1>
    %32 = migraphx.where %31, %11, %25 : <2x2x2x2x2xsi8, 4x0x2x0x1>, <2x2x2x2x2xf16, 0x0x0x0x0>, <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>

    // Sliding-window mask.
    %sliding_offset_bcast = migraphx.multibroadcast %sliding_offset {out_dyn_dims = [], out_lens = [2, 1]} : <1xsi32, 1> -> <2x1xsi32, 0x1>
    %window_start = migraphx.add %seq_len, %sliding_offset_bcast : <2x1xsi32, 1x1>, <2x1xsi32, 0x1> -> <2x1xsi32, 1x1>
    %window_starts = migraphx.multibroadcast %window_start {out_dyn_dims = [], out_lens = [2, 4]} : <2x1xsi32, 1x1> -> <2x4xsi32, 1x0>
    %window_pred = migraphx.greater %window_starts, %26 : <2x4xsi32, 1x0>, <2x4xsi32, 0x1> -> <2x4xsi32, 4x1>
    %window_i8 = migraphx.convert %window_pred {target_type = 0 : i64} : <2x4xsi32, 4x1> to <2x4xsi8, 4x1>
    %window_reshaped = migraphx.reshape %window_i8 {dims = [2, 1, 2, 1, 2]} : <2x4xsi8, 4x1> -> <2x1x2x1x2xsi8, 4x4x2x2x1>
    %window_mask = migraphx.multibroadcast %window_reshaped {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 2]} : <2x1x2x1x2xsi8, 4x4x2x2x1> -> <2x2x2x2x2xsi8, 4x0x2x0x1>
    %window_masked = migraphx.where %window_mask, %11, %32 : <2x2x2x2x2xsi8, 4x0x2x0x1>, <2x2x2x2x2xf16, 0x0x0x0x0>, <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>

    %33 = migraphx.convert %window_masked {target_type = 2 : i64} : <2x2x2x2x2xf16, 16x8x4x2x1> to <2x2x2x2x2xf32, 16x8x4x2x1>
    %34 = migraphx.reshape %33 {dims = [2, 2, 2, 2, 2]} : <2x2x2x2x2xf32, 16x8x4x2x1> -> <2x2x2x2x2xf32, 16x8x4x2x1>
    %35 = migraphx.reduce_max %34 {axes = [4]} : <2x2x2x2x2xf32, 16x8x4x2x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    %36 = migraphx.reshape %35 {dims = [2, 2, 2, 2, 1]} : <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    %37 = migraphx.multibroadcast %36 {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 2]} : <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x2xf32, 8x4x2x1x0>
    %38 = migraphx.sub %33, %37 : <2x2x2x2x2xf32, 16x8x4x2x1>, <2x2x2x2x2xf32, 8x4x2x1x0> -> <2x2x2x2x2xf32, 16x8x4x2x1>
    %39 = migraphx.exp %38 : <2x2x2x2x2xf32, 16x8x4x2x1> -> <2x2x2x2x2xf32, 16x8x4x2x1>
    %40 = migraphx.reshape %39 {dims = [2, 2, 2, 2, 2]} : <2x2x2x2x2xf32, 16x8x4x2x1> -> <2x2x2x2x2xf32, 16x8x4x2x1>
    %41 = migraphx.reduce_sum %40 {axes = [4]} : <2x2x2x2x2xf32, 16x8x4x2x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    %42 = migraphx.reshape %41 {dims = [2, 2, 2, 2, 1]} : <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    %43 = migraphx.multibroadcast %42 {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 2]} : <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x2xf32, 8x4x2x1x0>
    %44 = migraphx.div %39, %43 : <2x2x2x2x2xf32, 16x8x4x2x1>, <2x2x2x2x2xf32, 8x4x2x1x0> -> <2x2x2x2x2xf32, 16x8x4x2x1>
    %45 = migraphx.convert %44 {target_type = 1 : i64} : <2x2x2x2x2xf32, 16x8x4x2x1> to <2x2x2x2x2xf16, 16x8x4x2x1>
    %46 = migraphx.dot %45, %8 : <2x2x2x2x2xf16, 16x8x4x2x1>, <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>
    %47 = migraphx.transpose %46 {permutation = [0, 2, 3, 1, 4]} : <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x2x2xf16, 16x4x2x8x1>
    %48 = migraphx.reshape %47 {dims = [2, 2, 2, 4]} : <2x2x2x2x2xf16, 16x4x2x8x1> -> <2x2x2x4xf16, 16x8x4x1>
    %49 = migraphx.log %42 : <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    %50 = migraphx.add %36, %49 : <2x2x2x2x1xf32, 8x4x2x1x1>, <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    return %48, %50 : !migraphx.shaped<2x2x2x4xf16, 16x8x4x1>, !migraphx.shaped<2x2x2x2x1xf32, 8x4x2x1x1>
  }
}
