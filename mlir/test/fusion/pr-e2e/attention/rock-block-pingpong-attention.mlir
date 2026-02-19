// E2E accuracy test for block ping-pong scheduling on attention.
// Verifies that block ping-pong does not change numerical results.

// RUN: env ROCMLIR_ENABLE_BLOCK_PINGPONG=1 rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Simple attention pattern: Q @ K^T scaled -> softmax -> @ V
// This should result in 8 waves per block for block ping-pong to be active
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x8x64x128xf16, 65536x8192x128x1>, %arg1: !migraphx.shaped<1x8x64x128xf16, 65536x8192x128x1>, %arg2: !migraphx.shaped<1x8x64x128xf16, 65536x8192x128x1>) -> !migraphx.shaped<1x8x64x128xf16, 65536x8192x128x1> {
    %scale = migraphx.literal(dense<8.837890e-02> : tensor<1xf16>) : <1xf16, 1>
    
    // K^T: transpose K's last two dimensions
    %kt = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x8x64x128xf16, 65536x8192x128x1> -> <1x8x128x64xf16, 65536x8192x1x128>
    
    // Q @ K^T
    %qk = migraphx.dot %arg0, %kt : <1x8x64x128xf16, 65536x8192x128x1>, <1x8x128x64xf16, 65536x8192x1x128> -> <1x8x64x64xf16, 32768x4096x64x1>
    
    // Scale
    %scale_bc = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [1, 8, 64, 64]} : <1xf16, 1> -> <1x8x64x64xf16, 0x0x0x0>
    %qk_scaled = migraphx.mul %qk, %scale_bc : <1x8x64x64xf16, 32768x4096x64x1>, <1x8x64x64xf16, 0x0x0x0> -> <1x8x64x64xf16, 32768x4096x64x1>
    
    // Softmax: exp(x - max) / sum(exp(x - max))
    %qk_f32 = migraphx.convert %qk_scaled {target_type = 2 : i64} : <1x8x64x64xf16, 32768x4096x64x1> to <1x8x64x64xf32, 32768x4096x64x1>
    %max_val = migraphx.reduce_max %qk_f32 {axes = [3]} : <1x8x64x64xf32, 32768x4096x64x1> -> <1x8x64x1xf32, 512x64x1x1>
    %max_bc = migraphx.multibroadcast %max_val {out_dyn_dims = [], out_lens = [1, 8, 64, 64]} : <1x8x64x1xf32, 512x64x1x1> -> <1x8x64x64xf32, 512x64x1x0>
    %qk_sub = migraphx.sub %qk_f32, %max_bc : <1x8x64x64xf32, 32768x4096x64x1>, <1x8x64x64xf32, 512x64x1x0> -> <1x8x64x64xf32, 32768x4096x64x1>
    %qk_exp = migraphx.exp %qk_sub : <1x8x64x64xf32, 32768x4096x64x1> -> <1x8x64x64xf32, 32768x4096x64x1>
    %sum_val = migraphx.reduce_sum %qk_exp {axes = [3]} : <1x8x64x64xf32, 32768x4096x64x1> -> <1x8x64x1xf32, 512x64x1x1>
    %sum_bc = migraphx.multibroadcast %sum_val {out_dyn_dims = [], out_lens = [1, 8, 64, 64]} : <1x8x64x1xf32, 512x64x1x1> -> <1x8x64x64xf32, 512x64x1x0>
    %softmax = migraphx.div %qk_exp, %sum_bc : <1x8x64x64xf32, 32768x4096x64x1>, <1x8x64x64xf32, 512x64x1x0> -> <1x8x64x64xf32, 32768x4096x64x1>
    %softmax_f16 = migraphx.convert %softmax {target_type = 1 : i64} : <1x8x64x64xf32, 32768x4096x64x1> to <1x8x64x64xf16, 32768x4096x64x1>
    
    // @ V
    %out = migraphx.dot %softmax_f16, %arg2 : <1x8x64x64xf16, 32768x4096x64x1>, <1x8x64x128xf16, 65536x8192x128x1> -> <1x8x64x128xf16, 65536x8192x128x1>
    return %out : !migraphx.shaped<1x8x64x128xf16, 65536x8192x128x1>
  }
}
