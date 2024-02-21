// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_attention_wrapper -relDiff_threshold 0.02 -absDiff_threshold 0.02 -RMS_threshold 0.01  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]

module {
    func.func private @mlir_attention(%arg0: !migraphx.shaped<2x4096x640xf16, 2621440x640x1>, 
                                      %arg1: !migraphx.shaped<2x77x1280xf16, 98560x1280x1>, 
                                      %arg2: !migraphx.shaped<2x77x1280xf16, 98560x1280x1>) 
                                      -> !migraphx.shaped<20x4096x64xf16, 262144x64x1> 
                                      // attributes {arch = "gfx942:sramecc+:xnack-", kernel = "mixr", num_cu = 304 : i64} 
                                      {
    %0 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 0>
    %1 = migraphx.reshape %arg0 {dims = [2, 4096, 10, 64]} : <2x4096x640xf16, 2621440x640x1> -> <2x4096x10x64xf16, 2621440x640x64x1>
    %2 = migraphx.transpose %1 {permutation = [0, 2, 1, 3]} : <2x4096x10x64xf16, 2621440x640x64x1> -> <2x10x4096x64xf16, 2621440x64x640x1>
    %3 = migraphx.reshape %2 {dims = [20, 4096, 64]} : <2x10x4096x64xf16, 2621440x64x640x1> -> <20x4096x64xf16, 262144x64x1>
    %4 = migraphx.reshape %arg1 {dims = [2, 77, 2, 10, 64]} : <2x77x1280xf16, 98560x1280x1> -> <2x77x2x10x64xf16, 98560x1280x640x64x1>
    %5 = migraphx.transpose %4 {permutation = [2, 0, 3, 1, 4]} : <2x77x2x10x64xf16, 98560x1280x640x64x1> -> <2x2x10x77x64xf16, 640x98560x64x1280x1>
    %6 = migraphx.slice %5 {axes = [0], ends = [1], starts = [0]} : <2x2x10x77x64xf16, 640x98560x64x1280x1> -> <1x2x10x77x64xf16, 640x98560x64x1280x1>
    %7 = migraphx.reshape %6 {dims = [20, 77, 64]} : <1x2x10x77x64xf16, 640x98560x64x1280x1> -> <20x77x64xf16, 4928x64x1>
    %8 = migraphx.transpose %7 {permutation = [0, 2, 1]} : <20x77x64xf16, 4928x64x1> -> <20x64x77xf16, 4928x1x64>
    %9 = migraphx.dot %3, %8 : <20x4096x64xf16, 262144x64x1>, <20x64x77xf16, 4928x1x64> -> <20x4096x77xf16, 315392x77x1>
    %10 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [20, 4096, 77]} : <1xf16, 0> -> <20x4096x77xf16, 0x0x0>
    %11 = migraphx.mul %9, %10 : <20x4096x77xf16, 315392x77x1>, <20x4096x77xf16, 0x0x0> -> <20x4096x77xf16, 315392x77x1>
    %12 = migraphx.softmax %11 {axis = 2 : i64} : <20x4096x77xf16, 315392x77x1> -> <20x4096x77xf16, 315392x77x1>
    %13 = migraphx.reshape %arg2 {dims = [2, 77, 2, 10, 64]} : <2x77x1280xf16, 98560x1280x1> -> <2x77x2x10x64xf16, 98560x1280x640x64x1>
    %14 = migraphx.transpose %13 {permutation = [2, 0, 3, 1, 4]} : <2x77x2x10x64xf16, 98560x1280x640x64x1> -> <2x2x10x77x64xf16, 640x98560x64x1280x1>
    %15 = migraphx.slice %14 {axes = [0], ends = [2], starts = [1]} : <2x2x10x77x64xf16, 640x98560x64x1280x1> -> <1x2x10x77x64xf16, 640x98560x64x1280x1>
    %16 = migraphx.reshape %15 {dims = [20, 77, 64]} : <1x2x10x77x64xf16, 640x98560x64x1280x1> -> <20x77x64xf16, 4928x64x1>
    %17 = migraphx.dot %12, %16 : <20x4096x77xf16, 315392x77x1>, <20x77x64xf16, 4928x64x1> -> <20x4096x64xf16, 262144x64x1>
    return %17 : !migraphx.shaped<20x4096x64xf16, 262144x64x1>
  }
}
