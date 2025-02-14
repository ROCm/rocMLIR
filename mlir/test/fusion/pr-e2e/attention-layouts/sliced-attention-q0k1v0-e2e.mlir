// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -RMS_threshold=1e-2 -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full --debug-only=rock-gridwise-to-blockwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION

// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

// EMITKEY: -t f16 -transQ false -transK true -transV false -transO false -g 1 -seq_len_q 32 -seq_len_k 64 -head_dim_qk 16 -head_dim_v 8

// VECTORIZATION: qVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: qVectorLen: 4
// VECTORIZATION: kVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: kVectorLen: 4
// VECTORIZATION: vVectorDim: GemmDimension::MorN
// VECTORIZATION-NEXT: vVectorLen: 8

module {
  func.func @test(%arg0: !migraphx.shaped<1x32x32xf16, 1024x1x32>, %arg1: !migraphx.shaped<1x16x64xf16, 1024x1x16>, %arg2: !migraphx.shaped<1x8x64xf16, 512x1x8>) -> !migraphx.shaped<1x32x8xf16, 256x8x1> {
    %sliced0 = migraphx.slice %arg0 {axes = [1], ends = [16], starts = [0]} : <1x32x32xf16, 1024x1x32> -> <1x16x32xf16, 1024x1x32>
    %trans0 = migraphx.transpose %sliced0 {permutation = [0, 2, 1]} : <1x16x32xf16, 1024x1x32> -> <1x32x16xf16, 1024x32x1>
    %0 = migraphx.dot %trans0, %arg1: !migraphx.shaped<1x32x16xf16, 1024x32x1>, !migraphx.shaped<1x16x64xf16, 1024x1x16> -> !migraphx.shaped<1x32x64xf16, 2048x64x1>
    %1 = migraphx.softmax %0{axis = 2 : i64} : !migraphx.shaped<1x32x64xf16, 2048x64x1> -> !migraphx.shaped<1x32x64xf16, 2048x64x1>
    %trans2 = migraphx.transpose %arg2 {permutation = [0, 2, 1]} : <1x8x64xf16, 512x1x8> -> <1x64x8xf16, 512x8x1>
    %2 = migraphx.dot %1, %trans2: !migraphx.shaped<1x32x64xf16, 2048x64x1>, !migraphx.shaped<1x64x8xf16, 512x8x1> -> !migraphx.shaped<1x32x8xf16, 256x8x1>
    return %2 : !migraphx.shaped<1x32x8xf16, 256x8x1>
  }
}
