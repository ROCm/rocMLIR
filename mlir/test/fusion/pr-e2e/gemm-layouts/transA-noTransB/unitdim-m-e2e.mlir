// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -RMS_threshold=1e-2 -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch gfx1200 -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets gfx1200 | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full --debug-only=rock-gridwise-to-blockwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION

// CLONE: [1 1 1]

// EMITKEY: -t f16 -out_datatype f16 -transA true -transB false -g 2 -m 1 -n 640 -k 320

// VECTORIZATION: aVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: aVectorLen: 8
// VECTORIZATION: bVectorDim: GemmDimension::MorN
// VECTORIZATION-NEXT: bVectorLen: 4

module {
  func.func @test(%arg0: !migraphx.shaped<2x1x320xf16, 320x1x1>, %arg1: !migraphx.shaped<2x640x320xf16, 204800x1x640>, %arg2: !migraphx.shaped<2x64x10xf16, 0x10x1>) -> !migraphx.shaped<2x64x10xf16, 640x10x1> {
    %trans1 = migraphx.transpose %arg1 {permutation = [0, 2, 1]} : <2x640x320xf16, 204800x1x640> -> <2x320x640xf16, 204800x640x1>
    %2 = migraphx.dot %arg0, %trans1 : <2x1x320xf16, 320x1x1>, <2x320x640xf16, 204800x640x1> -> <2x1x640xf16, 640x640x1>
    %3 = migraphx.reshape %2 {dims = [2, 64, 10]} : <2x1x640xf16, 640x640x1> -> <2x64x10xf16, 640x10x1>
    %4 = migraphx.add %3, %arg2 : <2x64x10xf16, 640x10x1>, <2x64x10xf16, 0x10x1> -> <2x64x10xf16, 640x10x1>
    return %4 : !migraphx.shaped<2x64x10xf16, 640x10x1>
  }
}
