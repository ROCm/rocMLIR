// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -relDiff_threshold 0.09 -absDiff_threshold 1 -RMS_threshold 0.05 -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch gfx942 -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets gfx942 | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full --debug-only=rock-gridwise-to-blockwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]
// EMITKEY: convfp16 -F 1 -f N01GC -I 01NGC -O N01GC -n 1 -c 128 -H 80 -W 80 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1
// VECTORIZATION: aVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: aVectorLen: 8
// VECTORIZATION: bVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: bVectorLen: 8

module {
func.func @test(%arg0: !migraphx.shaped<1x256x80x80xf16, 1638400x1x20480x256>, %arg1: !migraphx.shaped<128x128x3x3xf16, 1152x1x384x128>) -> !migraphx.shaped<1x128x80x80xf16, 819200x1x10240x128> {
    %0 = migraphx.slice %arg0 {axes = [1], ends = [256], starts = [128]} : <1x256x80x80xf16, 1638400x1x20480x256> -> <1x128x80x80xf16, 1638400x1x20480x256>
    %1 = migraphx.sigmoid %0 : <1x128x80x80xf16, 1638400x1x20480x256> -> <1x128x80x80xf16, 819200x1x10240x128>
    %2 = migraphx.mul %0, %1 : <1x128x80x80xf16, 1638400x1x20480x256>, <1x128x80x80xf16, 819200x1x10240x128> -> <1x128x80x80xf16, 819200x1x10240x128>
    %3 = migraphx.convolution %2, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1], perf_config = "v3:64,64,8,16,16,4,1,1,2,1,1"} : <1x128x80x80xf16, 819200x1x10240x128>, <128x128x3x3xf16, 1152x1x384x128> -> <1x128x80x80xf16, 819200x1x10240x128>
    return %3 : !migraphx.shaped<1x128x80x80xf16, 819200x1x10240x128>
  }
}
