// The test case that was used to reproduce https://github.com/ROCmSoftwarePlatform/rocMLIR-internal/issues/940
// RUN: rocmlir-opt --migraphx-transform --canonicalize --migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -fut mlir_dot --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2

// CLONE: [1 1 1]
module {
  func.func @mlir_dot(%arg0: !migraphx.shaped<1x384x2304xf32, 884736x2304x1>, %arg1: !migraphx.shaped<1x384x2304xf32, 884736x2304x1>) -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1> {
    %0 = migraphx.literal (dense<1.250000e-01> : tensor<1xf32>) : !migraphx.shaped<1xf32, 0>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 12, 384, 384]} : !migraphx.shaped<1xf32, 0> -> !migraphx.shaped<1x12x384x384xf32, 0x0x0x0>
    %2 = migraphx.reshape %arg0 {dims = [1, 384, 36, 64]} : !migraphx.shaped<1x384x2304xf32, 884736x2304x1> -> !migraphx.shaped<1x384x36x64xf32, 884736x2304x64x1>
    %3 = migraphx.transpose %2 {permutation = [0, 2, 1, 3]} : !migraphx.shaped<1x384x36x64xf32, 884736x2304x64x1> -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
    %4 = migraphx.slice %3 {axes = [1], ends = [12], starts = [0]} : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> -> !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
    %5 = migraphx.reshape %arg1 {dims = [1, 384, 36, 64]} : !migraphx.shaped<1x384x2304xf32, 884736x2304x1> -> !migraphx.shaped<1x384x36x64xf32, 884736x2304x64x1>
    %6 = migraphx.transpose %5 {permutation = [0, 2, 1, 3]} : !migraphx.shaped<1x384x36x64xf32, 884736x2304x64x1> -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
    %7 = migraphx.slice %6 {axes = [1], ends = [24], starts = [12]} : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> -> !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
    %8 = migraphx.transpose %7 {permutation = [0, 1, 3, 2]} : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> -> !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1>
    %9 = migraphx.transpose %4 {permutation = [0, 1, 3, 2]} : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> -> !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1>
    %10 = migraphx.transpose %8 {permutation = [0, 1, 3, 2]} : !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1> -> !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
    %11 = migraphx.dot %10, %9 : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>, !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1> -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
    %12 = migraphx.transpose %11 {permutation = [0, 1, 3, 2]} : !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1> -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
    %13 = migraphx.mul %12, %1 : !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>, !migraphx.shaped<1x12x384x384xf32, 0x0x0x0> -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
    return %13 : !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
  }
}
