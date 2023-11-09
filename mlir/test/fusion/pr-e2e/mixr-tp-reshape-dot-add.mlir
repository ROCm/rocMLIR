// RUN: rocmlir-driver -kernel-pipeline migraphx %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_dot(%arg0: !migraphx.shaped<1x384x768xf32, 294912x768x1>, %arg1: !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>, %arg2: !migraphx.shaped<1x768x768xf32, 589824x768x1>) -> !migraphx.shaped<1x384x768xf32, 294912x768x1> {
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 768, 768]} : <1x768x768xf32, 589824x768x1> -> <1x768x768xf32, 589824x768x1>
    %1 = migraphx.transpose %arg1 {permutation = [0, 2, 1, 3]} : <1x12x384x64xf32, 294912x24576x64x1> -> <1x384x12x64xf32, 294912x768x64x1>
    %2 = migraphx.reshape %1 {dims = [1, 384, 768]} : <1x384x12x64xf32, 294912x768x64x1> -> <1x384x768xf32, 294912x768x1>
    %3 = migraphx.dot %2, %0 : <1x384x768xf32, 294912x768x1>, <1x768x768xf32, 589824x768x1> -> <1x384x768xf32, 294912x768x1>
    %4 = migraphx.add %3, %arg0 : <1x384x768xf32, 294912x768x1>, <1x384x768xf32, 294912x768x1> -> <1x384x768xf32, 294912x768x1>
    return %4 : !migraphx.shaped<1x384x768xf32, 294912x768x1>
  }
}
