// RUN: rocmlir-gen -fut mlir_conv_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_conv_gemm_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]

module {
  func.func @mlir_conv_gemm(%arg0: !migraphx.shaped<2x8x8x16xf16, 1024x8x1x64>, %arg1: !migraphx.shaped<16x16x3x3xf16, 144x1x48x16>, %arg2: !migraphx.shaped<2x16x6x6xf16, 0x1x0x0>, %arg3: !migraphx.shaped<2x16x6x6xf16, 0x1x0x0>, %arg4: !migraphx.shaped<1x16x32xf16, 0x1x0>) -> !migraphx.shaped<1x72x32xf16, 2304x32x1> {
    %transposed = migraphx.transpose %arg0 {permutation = [0, 3, 1, 2]} : <2x8x8x16xf16, 1024x8x1x64> -> <2x16x8x8xf16, 1024x64x8x1>
    %1 = migraphx.convolution %transposed, %arg1 {dilation = [2, 2], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x16x8x8xf16, 1024x64x8x1>, <16x16x3x3xf16, 144x1x48x16> -> <2x16x6x6xf16, 576x1x96x16>
    %2 = migraphx.transpose %1 {permutation = [0, 2, 3, 1]} : <2x16x6x6xf16, 576x1x96x16> -> <2x6x6x16xf16, 576x96x16x1>
    %3 = migraphx.reshape %2 {dims = [1, 72, 16]} : <2x6x6x16xf16, 576x96x16x1> -> <1x72x16xf16, 1152x16x1>
    
    %arg2tr = migraphx.transpose %arg2 {permutation = [0, 2, 3, 1]} : <2x16x6x6xf16, 0x1x0x0> -> <2x6x6x16xf16, 0x0x0x1>
    %arg2rs = migraphx.reshape %arg2tr {dims = [1, 72, 16]} : <2x6x6x16xf16, 0x0x0x1> -> <1x72x16xf16, 0x0x1>

    %arg3tr = migraphx.transpose %arg3 {permutation = [0, 2, 3, 1]} : <2x16x6x6xf16, 0x1x0x0> -> <2x6x6x16xf16, 0x0x0x1>
    %arg3rs = migraphx.reshape %arg3tr {dims = [1, 72, 16]} : <2x6x6x16xf16, 0x0x0x1> -> <1x72x16xf16, 0x0x1>

    %4 = migraphx.add %3, %arg2rs : <1x72x16xf16, 1152x16x1>, <1x72x16xf16, 0x0x1> -> <1x72x16xf16, 2048x16x1>
    %5 = migraphx.mul %4, %arg3rs : <1x72x16xf16, 2048x16x1>, <1x72x16xf16, 0x0x1> -> <1x72x16xf16, 2048x16x1>
    %8 = migraphx.dot %5, %arg4: <1x72x16xf16, 2048x16x1>, <1x16x32xf16, 0x1x0> -> <1x72x32xf16, 2304x32x1>
    return %8 : !migraphx.shaped<1x72x32xf16, 2304x32x1>
  }
}
