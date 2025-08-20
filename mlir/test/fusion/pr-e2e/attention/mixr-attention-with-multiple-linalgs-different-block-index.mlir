// this test is generates attention where firstGemm output index in preSoftmaxBody is at 2
// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x16x16xf16, 256x16x1>, 
                           %arg1: !migraphx.shaped<1x16x16xf16, 256x16x1>,
                           %arg2: !migraphx.shaped<1x16x16xf16, 256x16x1>,
                           %arg3: !migraphx.shaped<16x16xf16, 16x1>,
                           %arg4: !migraphx.shaped<1x16x16xf16, 256x16x1>) 
                           -> !migraphx.shaped<1x16x16xf16, 256x16x1> {
    // Q (query) tensor
    %q = migraphx.reshape %arg0 {dims = [1, 16, 16]} : <1x16x16xf16, 256x16x1> -> <1x16x16xf16, 256x16x1>
    
    // K (key) tensor - need to transpose for attention
    %k_reshaped = migraphx.reshape %arg1 {dims = [1, 16, 16]} : <1x16x16xf16, 256x16x1> -> <1x16x16xf16, 256x16x1>
    %k = migraphx.transpose %k_reshaped {permutation = [0, 2, 1]} : <1x16x16xf16, 256x16x1> -> <1x16x16xf16, 256x1x16>
    
    // V (value) tensor  
    %v = migraphx.reshape %arg2 {dims = [1, 16, 16]} : <1x16x16xf16, 256x16x1> -> <1x16x16xf16, 256x16x1>
    
    // Additional operands for elementwise operations
    %extra2 = migraphx.reshape %arg4 {dims = [1, 16, 16]} : <1x16x16xf16, 256x16x1> -> <1x16x16xf16, 256x16x1>
    
    // QK = Q * K^T
    %qk = migraphx.dot %q, %k : <1x16x16xf16, 256x16x1>, <1x16x16xf16, 256x1x16> -> <1x16x16xf16, 256x16x1>

    %reshape = migraphx.reshape %qk {dims = [16, 16]} : <1x16x16xf16, 256x16x1> -> <16x16xf16, 16x1> 
    // First elementwise operation: arg3 + QK
    %qk_plus_arg3 = migraphx.add %arg3, %reshape : <16x16xf16, 16x1>, <16x16xf16, 16x1> -> <16x16xf16, 16x1>

    %reshape_back = migraphx.reshape %qk_plus_arg3 {dims = [1, 16, 16]} : <16x16xf16, 16x1> -> <1x16x16xf16, 256x16x1> 
    // Second elementwise operation: (QK + arg3) * extra2
    %qk_elementwise = migraphx.mul %extra2, %reshape_back : <1x16x16xf16, 256x16x1>, <1x16x16xf16, 256x16x1> -> <1x16x16xf16, 256x16x1>
    // convert to f32 for softmax    
    %qk_elementwise_f32 = migraphx.convert %qk_elementwise : <1x16x16xf16, 256x16x1> to <1x16x16xf32, 256x16x1>
    // Softmax operations
    // 1. Reduce max
    %max_val = migraphx.reduce_max %qk_elementwise_f32 {axes = [2]} : <1x16x16xf32, 256x16x1> -> <1x16x1xf32, 16x1x1>
    %max_broadcast = migraphx.multibroadcast %max_val {out_lens = [1, 16, 16]} : <1x16x1xf32, 16x1x1> -> <1x16x16xf32, 16x1x0>
    
    // 2. Subtract max
    %sub_max = migraphx.sub %qk_elementwise_f32, %max_broadcast : <1x16x16xf32, 256x16x1>, <1x16x16xf32, 16x1x0> -> <1x16x16xf32, 256x16x1>
    
    // 3. Exp
    %exp_val = migraphx.exp %sub_max : <1x16x16xf32, 256x16x1> -> <1x16x16xf32, 256x16x1>
    
    // 4. Reduce sum
    %sum_val = migraphx.reduce_sum %exp_val {axes = [2]} : <1x16x16xf32, 256x16x1> -> <1x16x1xf32, 16x1x1>
    %sum_broadcast = migraphx.multibroadcast %sum_val {out_lens = [1, 16, 16]} : <1x16x1xf32, 16x1x1> -> <1x16x16xf32, 16x1x0>
    
    // 5. Divide (normalize)
    %softmax_out = migraphx.div %exp_val, %sum_broadcast : <1x16x16xf32, 256x16x1>, <1x16x16xf32, 16x1x0> -> <1x16x16xf32, 256x16x1>
    // Convert softmax output to f16
    %softmax_out_f16 = migraphx.convert %softmax_out : <1x16x16xf32, 256x16x1> to <1x16x16xf16, 256x16x1>
    
    // Attention output: softmax(QK) * V
    %attention_out = migraphx.dot %softmax_out_f16, %v : <1x16x16xf16, 256x16x1>, <1x16x16xf16, 256x16x1> -> <1x16x16xf16, 256x16x1>
    
    return %attention_out : !migraphx.shaped<1x16x16xf16, 256x16x1>
  }
}
