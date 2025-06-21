// RUN: rocmlir-gen -fut mlir_convolution_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_convolution_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
func.func private @mlir_convolution_add(%arg0: !migraphx.shaped<1x64x224xf32, 0x1x0>, %arg1: !migraphx.shaped<1x3x224xf32, 672x224x1>, %arg2: !migraphx.shaped<64x3x7xf32, 21x7x1>) -> !migraphx.shaped<1x64x224xf32, 14336x224x1> {
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [1], group = 1 : i64, padding = [3, 3], padding_mode = 0 : i64, stride = [1]} : <1x3x224xf32, 672x224x1>, <64x3x7xf32, 21x7x1> -> <1x64x224xf32, 14336x224x1>
  %1 = migraphx.add %0, %arg0 : <1x64x224xf32, 14336x224x1>, <1x64x224xf32, 0x1x0> -> <1x64x224xf32, 14336x224x1>
  return %1 : !migraphx.shaped<1x64x224xf32, 14336x224x1>
}
