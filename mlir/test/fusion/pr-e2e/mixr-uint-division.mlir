// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut migraphx_div_ui32 --arch %arch --clone-harness - | rocmlir-driver -host-pipeline=highlevel | rocmlir-gen -ph -fut migraphx_div_ui32_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {

  func.func @migraphx_div_ui32(%arg0: !migraphx.shaped<1x2048x1x1xf32, 2048x1x1x1>, %arg1: !migraphx.shaped<1x2048x1x1xf32, 2048x1x1x1>, %arg2: !migraphx.shaped<1x2048x7x7xf32, 100352x49x7x1>, %arg3: !migraphx.shaped<1x2048x1x1xf32, 2048x1x1x1>, %arg4: !migraphx.shaped<1x1024x14x14xf32, 200704x196x14x1>, %arg5: !migraphx.shaped<2048x1024x1x1xf32, 1024x1x1x1>) -> !migraphx.shaped<1x2048x7x7xui32, 100352x49x7x1> {
    %0 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 2048, 7, 7]} : <1x2048x1x1xf32, 2048x1x1x1> -> <1x2048x7x7xf32, 0x1x0x0>
    %1 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [1, 2048, 7, 7]} : <1x2048x1x1xf32, 2048x1x1x1> -> <1x2048x7x7xf32, 0x1x0x0>
    %2 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [1, 2048, 7, 7]} : <1x2048x1x1xf32, 2048x1x1x1> -> <1x2048x7x7xf32, 0x1x0x0>
    %3 = migraphx.convolution %arg4, %arg5 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2]} : <1x1024x14x14xf32, 200704x196x14x1>, <2048x1024x1x1xf32, 1024x1x1x1> -> <1x2048x7x7xf32, 100352x49x7x1>
    %4 = migraphx.mul %2, %3 : <1x2048x7x7xf32, 0x1x0x0>, <1x2048x7x7xf32, 100352x49x7x1> -> <1x2048x7x7xf32, 100352x49x7x1>
    %5 = migraphx.mul %1, %4 : <1x2048x7x7xf32, 0x1x0x0>, <1x2048x7x7xf32, 100352x49x7x1> -> <1x2048x7x7xf32, 100352x49x7x1>
    %6 = migraphx.mul %2, %arg2 : <1x2048x7x7xf32, 0x1x0x0>, <1x2048x7x7xf32, 100352x49x7x1> -> <1x2048x7x7xf32, 100352x49x7x1>
    %7 = migraphx.mul %1, %6 : <1x2048x7x7xf32, 0x1x0x0>, <1x2048x7x7xf32, 100352x49x7x1> -> <1x2048x7x7xf32, 100352x49x7x1>
    %8 = migraphx.add %7, %5 : <1x2048x7x7xf32, 100352x49x7x1>, <1x2048x7x7xf32, 100352x49x7x1> -> <1x2048x7x7xf32, 100352x49x7x1>
    %9 = migraphx.add %8, %0 : <1x2048x7x7xf32, 100352x49x7x1>, <1x2048x7x7xf32, 0x1x0x0> -> <1x2048x7x7xf32, 100352x49x7x1>
    %10 = migraphx.literal (dense<[2]> : tensor<1xui32>) : <1xui32, 0>
    %11 = migraphx.multibroadcast %10 {out_dyn_dims = [], out_lens = [1, 2048, 7, 7]} : <1xui32, 0> -> <1x2048x7x7xui32, 0x0x0x0>
    %12 = migraphx.relu %9 : <1x2048x7x7xf32, 100352x49x7x1> -> <1x2048x7x7xf32, 100352x49x7x1>
    %13 = migraphx.convert %12 : <1x2048x7x7xf32, 100352x49x7x1> to <1x2048x7x7xui32, 100352x49x7x1>
    %14 = migraphx.div %13, %11 : <1x2048x7x7xui32, 100352x49x7x1>, <1x2048x7x7xui32, 0x0x0x0> -> <1x2048x7x7xui32, 100352x49x7x1>
    return %14 : !migraphx.shaped<1x2048x7x7xui32, 100352x49x7x1>
  }
}
