// RUN: rocmlir-gen -fut mlir_fusion --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_fusion_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
func.func @mlir_fusion(%arg0: !migraphx.shaped<1x5xf32, 5x1>, %arg1: !migraphx.shaped<2x5xf32, 5x1>, %arg2: !migraphx.shaped<2x5xf32, 5x1>, %arg3: !migraphx.shaped<2x5xf32, 5x1>, %arg4: !migraphx.shaped<2x5xf32, 5x1>, %arg5: !migraphx.shaped<2x5xf32, 5x1>, %arg6: !migraphx.shaped<2x5xf32, 5x1>, %arg7: !migraphx.shaped<15x5xf32, 5x1>) -> !migraphx.shaped<2x5xf32, 5x1> {
  %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [2, 5]} : <1x5xf32, 5x1> -> <2x5xf32, 0x1>
  %1 = migraphx.transpose %arg7 {permutation = [1, 0]} : <15x5xf32, 5x1> -> <5x15xf32, 15x1>
  %2 = migraphx.slice %1 {axes = [1], ends = [15], starts = [10]} : <5x15xf32, 15x1> -> <5x5xf32, 5x1>
  %3 = migraphx.dot %arg6, %2 : <2x5xf32, 5x1>, <5x5xf32, 5x1> -> <2x5xf32, 5x1>
  %4 = migraphx.add %3, %0 : <2x5xf32, 5x1>, <2x5xf32, 0x1> -> <2x5xf32, 5x1>
  %5 = migraphx.add %arg1, %4 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
  %6 = migraphx.tanh %5 : <2x5xf32, 5x1> -> <2x5xf32, 5x1>
  %7 = migraphx.add %arg2, %arg3 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
  %8 = migraphx.sigmoid %7 : <2x5xf32, 5x1> -> <2x5xf32, 5x1>
  %9 = migraphx.sub %arg4, %8 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
  %10 = migraphx.mul %9, %6 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
  %11 = migraphx.mul %8, %arg5 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
  %12 = migraphx.add %10, %11 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
  return %12 : !migraphx.shaped<2x5xf32, 5x1>
}
