// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch gfx1200 -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets gfx1200 | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full --debug-only=rock-gridwise-to-blockwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION

// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

// EMITKEY: convfp16 -F 1 -f NGC01 -I N01GC -O NGC01 -n 64 -c 80 -H 20 -W 20 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1

// VECTORIZATION: aVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: aVectorLen: 8
// VECTORIZATION: bVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: bVectorLen: 8

module {
  func.func @test(%arg0: !migraphx.shaped<80x80x1x1xf16, 80x1x1x1>, %arg1: !migraphx.shaped<64x80x20x20xf16, 32000x1x1600x80>, %arg2: !migraphx.shaped<64x80x20x20xf16, 0x1x0x0>) -> !migraphx.shaped<64x80x20x20xf16, 32000x1x1600x80> {
    %0 = migraphx.convolution %arg1, %arg0 {perf_config="v3:128,128,4,128,16,8,1,1,2,1,1", dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <64x80x20x20xf16, 32000x1x1600x80>, <80x80x1x1xf16, 80x1x1x1> -> <64x80x20x20xf16, 32000x1x1600x80>
    %1 = migraphx.add %0, %arg2 : <64x80x20x20xf16, 32000x1x1600x80>, <64x80x20x20xf16, 0x1x0x0> -> <64x80x20x20xf16, 32000x1x1600x80>
    %2 = migraphx.sigmoid %1 : <64x80x20x20xf16, 32000x1x1600x80> -> <64x80x20x20xf16, 32000x1x1600x80>
    %3 = migraphx.mul %1, %2 : <64x80x20x20xf16, 32000x1x1600x80>, <64x80x20x20xf16, 32000x1x1600x80> -> <64x80x20x20xf16, 32000x1x1600x80>
    return %3 : !migraphx.shaped<64x80x20x20xf16, 32000x1x1600x80>
  }
}
