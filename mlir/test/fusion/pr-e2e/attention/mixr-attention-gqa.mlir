// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.000004  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func private @mlir_attention(%v: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>, 
                            %q: !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>, 
                            %k: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>) 
                            -> (!migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>) {
    %vbroadcast = migraphx.multibroadcast %v {out_dyn_dims = [], out_lens = [2, 2, 2, 32, 32]} : <2x2x32x32xf32, 2048x1024x32x1> -> <2x2x2x32x32xf32, 2048x1024x0x32x1>
    %vreshaped = migraphx.reshape %vbroadcast {dims = [2, 4, 32, 32]} : <2x2x2x32x32xf32, 2048x1024x0x32x1> -> <2x4x32x32xf32, 2048x1024x32x1>
    %kbroadcast = migraphx.multibroadcast %k {out_dyn_dims = [], out_lens = [2, 2, 2, 32, 32]} : <2x2x32x32xf32, 2048x1024x32x1> -> <2x2x2x32x32xf32, 2048x1024x0x32x1>
    %kreshaped = migraphx.reshape %kbroadcast {dims = [2, 4, 32, 32]} : <2x2x2x32x32xf32, 2048x1024x0x32x1> -> <2x4x32x32xf32, 2048x1024x32x1>
    %kt = migraphx.transpose %kreshaped {permutation = [0, 1, 3, 2]} : <2x4x32x32xf32, 2048x1024x32x1> -> <2x4x32x32xf32, 2048x1024x32x1>
    %qk = migraphx.dot %q, %kt : <2x4x32x32xf32, 4096x1024x32x1>, <2x4x32x32xf32, 2048x1024x32x1> -> <2x4x32x32xf32, 4096x1024x32x1>
    %att = migraphx.softmax %qk {axis = 3 : i64} : <2x4x32x32xf32, 4096x1024x32x1> -> <2x4x32x32xf32, 4096x1024x32x1>
    %res = migraphx.dot %att, %vreshaped : <2x4x32x32xf32, 4096x1024x32x1>, <2x4x32x32xf32, 2048x1024x32x1> -> <2x4x32x32xf32, 4096x1024x32x1>
    return %res : !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>
  }
}
