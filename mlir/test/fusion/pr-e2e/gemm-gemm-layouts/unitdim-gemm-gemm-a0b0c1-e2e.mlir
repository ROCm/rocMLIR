// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch gfx942:sramecc+:xnack- -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets gfx942:sramecc+:xnack- | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full --debug-only=rock-gridwise-to-blockwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION

// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

// EMITKEY: -t f32 -transA false -transB false -transC true -transO false -g 1 -m 32 -n 64 -k 1 -gemmO 8

// VECTORIZATION: qVectorDim: GemmDimension::MorN
// VECTORIZATION-NEXT: qVectorLen: 4
// VECTORIZATION: kVectorDim: GemmDimension::MorN
// VECTORIZATION-NEXT: kVectorLen: 4
// VECTORIZATION: vVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: vVectorLen: 4

module {
  func.func @test(%arg0: !migraphx.shaped<1x32x1xf32, 32x1x1>, %arg1: !migraphx.shaped<1x1x64xf32, 64x1x1>, %arg2: !migraphx.shaped<1x64x8xf32, 512x1x64>) -> !migraphx.shaped<1x32x8xf32, 256x8x1> {
    %0 = migraphx.dot %arg0, %arg1: !migraphx.shaped<1x32x1xf32, 32x1x1>, !migraphx.shaped<1x1x64xf32, 64x1x1> -> !migraphx.shaped<1x32x64xf32, 2048x64x1>
    %2 = migraphx.dot %0, %arg2: !migraphx.shaped<1x32x64xf32, 2048x64x1>, !migraphx.shaped<1x64x8xf32, 512x1x64> -> !migraphx.shaped<1x32x8xf32, 256x8x1>
    return %2 : !migraphx.shaped<1x32x8xf32, 256x8x1>
  }
}
