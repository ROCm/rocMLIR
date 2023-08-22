// RUN: rocmlir-driver -kernel-pipeline migraphx %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_dot(%arg0: !migraphx.shaped<1x1x1x1xf32, 1x1x1x1>, %arg1: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>, %arg2: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>) -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1> {
    %0 = migraphx.literal (dense<1.250000e-01> : tensor<1xf32>) : !migraphx.shaped<1xf32, 1>
    %1 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [1, 12, 384, 384]} : !migraphx.shaped<1x1x1x1xf32, 1x1x1x1> -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
    %2 = migraphx.transpose %arg2 {permutation = [0, 1, 3, 2]} : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> -> !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1>
    %3 = migraphx.dot %arg1, %2 : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>, !migraphx.shaped<1x12x64x384xf32, 294912x24576x384x1> -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
    %4 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 12, 384, 384]} : !migraphx.shaped<1xf32, 1> -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
    %5 = migraphx.mul %3, %4 : !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>, !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1> -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
    %6 = migraphx.add %5, %1 : !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>, !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1> -> !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
    return %6 : !migraphx.shaped<1x12x384x384xf32, 1769472x147456x384x1>
  }
}
