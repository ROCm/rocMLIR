// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch gfx942:sramecc+:xnack- -fut test %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets gfx942:sramecc+:xnack- | rocmlir-gen -ph -verifier clone -fut test_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full --debug-only=rock-gridwise-to-blockwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION

// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

// EMITKEY: -t f16 -f GN01C -I N01GC -transC false -transO false -n 2 -c 16 -H 8 -W 8 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -gemmO 32

// VECTORIZATION: qVectorDim: GemmDimension::MorN
// VECTORIZATION-NEXT: qVectorLen: 1
// VECTORIZATION: kVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: kVectorLen: 2
// VECTORIZATION: vVectorDim: GemmDimension::K
// VECTORIZATION-NEXT: vVectorLen: 2

module {
  func.func @test(%arg0: !migraphx.shaped<2x8x8x32xf16, 2048x8x1x64>, %arg1: !migraphx.shaped<16x16x3x3xf16, 144x1x48x16>, %arg2: !migraphx.shaped<2x16x8x8xf16, 0x1x0x0>, %arg3: !migraphx.shaped<2x16x8x8xf16, 0x1x0x0>, %arg4: !migraphx.shaped<1x16x32xf16, 0x1x0>) -> !migraphx.shaped<1x128x32xf16, 4096x32x1> {
    %0 = migraphx.slice %arg0 {axes = [3], ends = [32], starts = [16]} : <2x8x8x32xf16, 2048x8x1x64> -> <2x8x8x16xf16, 2048x8x1x64>
    %transposed = migraphx.transpose %0 {permutation = [0, 3, 1, 2]} : <2x8x8x16xf16, 2048x8x1x64> -> <2x16x8x8xf16, 2048x64x8x1>
    %1 = migraphx.convolution %transposed, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x16x8x8xf16, 2048x64x8x1>, <16x16x3x3xf16, 144x1x48x16> -> <2x16x8x8xf16, 2048x1x128x16>
    %2 = migraphx.transpose %1 {permutation = [0, 2, 3, 1]} : <2x16x8x8xf16, 2048x1x128x16> -> <2x8x8x16xf16, 2048x128x16x1>
    %3 = migraphx.reshape %2 {dims = [1, 128, 16]} : <2x8x8x16xf16, 2048x128x16x1> -> <1x128x16xf16, 2048x16x1>
    
    %arg2tr = migraphx.transpose %arg2 {permutation = [0, 2, 3, 1]} : <2x16x8x8xf16, 0x1x0x0> -> <2x8x8x16xf16, 0x0x0x1>
    %arg2rs = migraphx.reshape %arg2tr {dims = [1, 128, 16]} : <2x8x8x16xf16, 0x0x0x1> -> <1x128x16xf16, 0x0x1>

    %arg3tr = migraphx.transpose %arg3 {permutation = [0, 2, 3, 1]} : <2x16x8x8xf16, 0x1x0x0> -> <2x8x8x16xf16, 0x0x0x1>
    %arg3rs = migraphx.reshape %arg3tr {dims = [1, 128, 16]} : <2x8x8x16xf16, 0x0x0x1> -> <1x128x16xf16, 0x0x1>

    %4 = migraphx.add %3, %arg2rs : <1x128x16xf16, 2048x16x1>, <1x128x16xf16, 0x0x1> -> <1x128x16xf16, 2048x16x1>
    %5 = migraphx.mul %4, %arg3rs : <1x128x16xf16, 2048x16x1>, <1x128x16xf16, 0x0x1> -> <1x128x16xf16, 2048x16x1>
    %8 = migraphx.dot %5, %arg4: <1x128x16xf16, 2048x16x1>, <1x16x32xf16, 0x1x0> -> <1x128x32xf16, 4096x32x1>
    return %8 : !migraphx.shaped<1x128x32xf16, 4096x32x1>
  }
}
