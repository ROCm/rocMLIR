// E2E accuracy test for block ping-pong scheduling on GEMM.
// Verifies that block ping-pong does not change numerical results.

// RUN: env ROCMLIR_ENABLE_BLOCK_PINGPONG=1 rocmlir-gen -fut test --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut test_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

// GEMM with shapes that result in 8 waves per block (block_size=512, waveSize=64)
// Using f16 for typical GEMM workload on MI GPUs
module {
  func.func @test(%arg0: !migraphx.shaped<2x512x256xf16, 131072x256x1>, %arg1: !migraphx.shaped<2x256x512xf16, 131072x512x1>) -> !migraphx.shaped<2x512x512xf16, 262144x512x1> attributes {kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 : <2x512x256xf16, 131072x256x1>, <2x256x512xf16, 131072x512x1> -> <2x512x512xf16, 262144x512x1>
    return %0 : !migraphx.shaped<2x512x512xf16, 262144x512x1>
  }
}
