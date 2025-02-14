// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full --debug-only=rock-gridwise-to-blockwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION

// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

// EMITKEY: convfp16 -F 1 -f GN01C -I NG0C1 -O NGC01 -n 2 -c 32 -H 160 -W 160 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1

// VECTORIZATION: aVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: aVectorLen: 1
// VECTORIZATION: bVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: bVectorLen: 1

module {
  func.func @test(%arg0: !migraphx.shaped<16x32x3x3xf16, 288x1x96x32>, %arg1: !migraphx.shaped<2x2x1x160x160xf16, 51200x25600x1x160x1>, %arg2: !migraphx.shaped<2x16x160x160xf16, 0x1x0x0>) -> !migraphx.shaped<2x16x160x160xf16, 409600x1x2560x16> {
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [2, 2, 16, 160, 160]} : <2x2x1x160x160xf16, 51200x25600x1x160x1> -> <2x2x16x160x160xf16, 51200x25600x1x160x1>
    %1 = migraphx.reshape %0 {dims = [2, 32, 160, 160]} : <2x2x16x160x160xf16, 51200x25600x1x160x1> -> <2x32x160x160xf16, 819200x1x2560x16>
    %2 = migraphx.convolution %1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x32x160x160xf16, 819200x1x2560x16>, <16x32x3x3xf16, 288x1x96x32> -> <2x16x160x160xf16, 409600x1x2560x16>
    %3 = migraphx.add %2, %arg2 : <2x16x160x160xf16, 409600x1x2560x16>, <2x16x160x160xf16, 0x1x0x0> -> <2x16x160x160xf16, 409600x1x2560x16>
    %4 = migraphx.sigmoid %3 : <2x16x160x160xf16, 409600x1x2560x16> -> <2x16x160x160xf16, 409600x1x2560x16>
    %5 = migraphx.mul %3, %4 : <2x16x160x160xf16, 409600x1x2560x16>, <2x16x160x160xf16, 409600x1x2560x16> -> <2x16x160x160xf16, 409600x1x2560x16>
    return %5 : !migraphx.shaped<2x16x160x160xf16, 409600x1x2560x16>
  }
}
