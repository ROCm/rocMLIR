// RUN: rocmlir-gen -fut mlir_conv_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_conv_gemm_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_conv_gemm(%arg0: !migraphx.shaped<2x8x8x16xf32, 1024x8x1x64>, %arg1: !migraphx.shaped<16x16x3x3xf32, 144x1x48x16>, %arg2: !migraphx.shaped<1x16x32xf32, 0x1x0>) -> !migraphx.shaped<1x128x32xf32, 4096x32x1> {
    %transposed = migraphx.transpose %arg0 {permutation = [0, 3, 1, 2]} : <2x8x8x16xf32, 1024x8x1x64> -> <2x16x8x8xf32, 1024x64x8x1>
    %1 = migraphx.convolution %transposed, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x16x8x8xf32, 1024x64x8x1>, <16x16x3x3xf32, 144x1x48x16> -> <2x16x8x8xf32, 2048x1x128x16>
    %2 = migraphx.transpose %1 {permutation = [0, 2, 3, 1]} : <2x16x8x8xf32, 2048x1x128x16> -> <2x8x8x16xf32, 2048x128x16x1>
    %3 = migraphx.reshape %2 {dims = [1, 128, 16]} : <2x8x8x16xf32, 2048x128x16x1> -> <1x128x16xf32, 2048x16x1>
    %4 = migraphx.dot %3, %arg2 {perf_config="attn:v2:64,128,32,16,32,16,4,4,1,2,1"} : <1x128x16xf32, 2048x16x1>, <1x16x32xf32, 0x1x0> -> <1x128x32xf32, 4096x32x1>
    return %4 : !migraphx.shaped<1x128x32xf32, 4096x32x1>
  }
}
