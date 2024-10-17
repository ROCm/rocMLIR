// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut mlir_unpack_int4_1645 --arch %arch --clone-harness - | rocmlir-driver -host-pipeline=highlevel | rocmlir-gen -ph -fut mlir_unpack_int4_1645_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
// COM: Runs the MIGraphX pipeline first to rewrite out the int4
module {
  func.func @mlir_unpack_int4_1645(%arg0: !migraphx.shaped<2x2xi8, 2x1>, %arg1: !migraphx.shaped<2x2x1x1x1x1xf16, 2x1x1x1x1x1>, %arg2: !migraphx.shaped<2x1xi8, 1x1>, %arg3: !migraphx.shaped<2x4xf16, 4x1>) -> !migraphx.shaped<4x4xf16, 4x1> // attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr", num_cu = 110 : i64} 
  {
    %0 = migraphx.unpack %arg0 {axis = 1 : i64, isUnsigned = false} : <2x2xi8, 2x1> -> <2x4xi8, 4x1>
    %1 = migraphx.unpack %arg2 {axis = 1 : i64, isUnsigned = false} : <2x1xi8, 1x1> -> <2x2xi8, 2x1>
    %2 = migraphx.reshape %arg1 {dims = [2, 2, 1, 1, 1, 1, 1]} : <2x2x1x1x1x1xf16, 2x1x1x1x1x1> -> <2x2x1x1x1x1x1xf16, 2x1x1x1x1x1x1>
    %3 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [2, 2, 1, 1, 1, 1, 2]} : <2x2x1x1x1x1x1xf16, 2x1x1x1x1x1x1> -> <2x2x1x1x1x1x2xf16, 2x1x1x1x1x1x0>
    %4 = migraphx.reshape %3 {dims = [2, 4]} : <2x2x1x1x1x1x2xf16, 2x1x1x1x1x1x0> -> <2x4xf16, 4x1>
    %5 = migraphx.reshape %1 {dims = [2, 2, 1]} : <2x2xi8, 2x1> -> <2x2x1xi8, 2x1x1>
    %6 = migraphx.multibroadcast %5 {out_dyn_dims = [], out_lens = [2, 2, 2]} : <2x2x1xi8, 2x1x1> -> <2x2x2xi8, 2x1x0>
    %7 = migraphx.reshape %6 {dims = [2, 4]} : <2x2x2xi8, 2x1x0> -> <2x4xi8, 4x1>
    %8 = migraphx.dequantizelinear %0, %4, %7 : <2x4xi8, 4x1>, <2x4xf16, 4x1>, !migraphx.shaped<2x4xi8, 4x1> -> <2x4xf16, 4x1>
    %9 = migraphx.transpose %8 {permutation = [1, 0]} : <2x4xf16, 4x1> -> <4x2xf16, 1x4>
    %10 = migraphx.dot %9, %arg3 : <4x2xf16, 1x4>, <2x4xf16, 4x1> -> <4x4xf16, 4x1>
    return %10 : !migraphx.shaped<4x4xf16, 4x1>
  }
}
