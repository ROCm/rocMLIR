// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK: [1 1 1]
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1>, %arg1: !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1>, %arg2: !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1>) -> !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1> attributes {rock.kernel} {
    %scale = migraphx.literal(dense<3.535160e-01> : tensor<1xf16>) : <1xf16, 1>
    %neg10k = migraphx.literal(dense<-1.000000e+04> : tensor<1xf16>) : <1xf16, 1>
    %mask = migraphx.literal(dense<[[1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]]> : tensor<8x8xsi8>) : <8x8xsi8, 8x1>
    %kt = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x2x8x64xf16, 1024x512x64x1> -> <1x2x64x8xf16, 1024x512x1x64>
    %qk = migraphx.dot %arg0, %kt : <1x2x8x64xf16, 1024x512x64x1>, <1x2x64x8xf16, 1024x512x1x64> -> <1x2x8x8xf16, 128x64x8x1>
    %scale_bcast = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [1, 2, 8, 8]} : <1xf16, 1> -> <1x2x8x8xf16, 0x0x0x0>
    %scaled = migraphx.mul %qk, %scale_bcast : <1x2x8x8xf16, 128x64x8x1>, <1x2x8x8xf16, 0x0x0x0> -> <1x2x8x8xf16, 128x64x8x1>
    %mask_bcast = migraphx.multibroadcast %mask {out_dyn_dims = [], out_lens = [1, 2, 8, 8]} : <8x8xsi8, 8x1> -> <1x2x8x8xsi8, 0x0x8x1>
    %neg10k_bcast = migraphx.multibroadcast %neg10k {out_dyn_dims = [], out_lens = [1, 2, 8, 8]} : <1xf16, 1> -> <1x2x8x8xf16, 0x0x0x0>
    %masked = migraphx.where %mask_bcast, %scaled, %neg10k_bcast : <1x2x8x8xsi8, 0x0x8x1>, <1x2x8x8xf16, 128x64x8x1>, <1x2x8x8xf16, 0x0x0x0> -> <1x2x8x8xf16, 128x64x8x1>
    %sm = migraphx.softmax %masked {axis = 3 : i64} : <1x2x8x8xf16, 128x64x8x1> -> <1x2x8x8xf16, 128x64x8x1>
    %out = migraphx.dot %sm, %arg2 : <1x2x8x8xf16, 128x64x8x1>, <1x2x8x64xf16, 1024x512x64x1> -> <1x2x8x64xf16, 1024x512x64x1>
    return %out : !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1>
  }
}
