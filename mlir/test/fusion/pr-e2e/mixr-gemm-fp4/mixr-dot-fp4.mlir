// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut mlir_dot --arch %arch --clone-harness - | rocmlir-driver -host-pipeline=highlevel -kernel-pipeline=highlevel | rocmlir-gen -ph -fut mlir_dot_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
func.func private @mlir_dot(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1>  {
  %0 = migraphx.dot %arg0, %arg1 : <1x16x512xf4E2M1FN, 8192x512x1>, <1x512x16xf4E2M1FN, 8192x16x1> -> <1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}
