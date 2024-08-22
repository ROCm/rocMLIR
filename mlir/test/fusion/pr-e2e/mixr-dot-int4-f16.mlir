// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut mlir_unpack_dequantizelinear_dot --arch %arch --clone-harness -  | FileCheck %s --check-prefix=HASINT4
// HASINT4: mhal.launch
// HASINT4-SAME: tensor<64xi4>

// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut mlir_unpack_dequantizelinear_dot --arch %arch --clone-harness - | rocmlir-driver -host-pipeline=highlevel | rocmlir-gen -ph -fut mlir_unpack_dequantizelinear_dot_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
// COM: Runs the MIGraphX pipeline first to rewrite out the int4
func.func private @mlir_unpack_dequantizelinear_dot(%arg0: !migraphx.shaped<1x4x8xi8, 32x8x1>, %arg1: !migraphx.shaped<1x16x4xf16, 64x4x1>) -> !migraphx.shaped<1x4x4xf16, 16x4x1>  {
  %0 = migraphx.literal (dense<[0.25]> : tensor<1xf16>) : <1xf16, 0>
  %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 5, 16]} : <1xf16, 0> -> <1x4x16xf16, 0x0x0>
  %2 = migraphx.unpack %arg0 {axis = 2 : i64, isUnsigned = false} : <1x4x8xi8, 32x8x1> -> <1x4x16xi8, 64x16x1>
  %3 = migraphx.dequantizelinear %2, %1 : <1x4x16xi8, 64x16x1>, <1x4x16xf16, 0x0x0> -> <1x4x16xf16, 64x16x1>
  %4 = migraphx.dot %3, %arg1 : <1x4x16xf16, 64x16x1>, <1x16x4xf16, 64x4x1> -> <1x4x4xf16, 16x4x1>
  return %4 : !migraphx.shaped<1x4x4xf16, 16x4x1>
}
