// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut mlir_dequantizelinear_convolution_quantizelinear --arch %arch --clone-harness - | rocmlir-driver -host-pipeline=highlevel | rocmlir-gen -print-results -ph -fut mlir_dequantizelinear_convolution_quantizelinear_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [49] strides = [1] data =
// CHECK-NEXT: [-4, -1, -6, -5, -5, -1, -13, -5, -6, -1, -10, -4, -7, -2, -5, -1, -1, -14, -1, -3, -15, -2, -9, -12, -13, -12, -6, -5, -1, -13, -5, -13, -1, -5, -9, -5, -8, -15, -11, -12, -3, -11, -15, -11, -5, -9, -11, -1, -11]
// COM: tests fail is they have no arguments, that's why we have %dummy
// COM: Note that values are negative because they are printed as signed integers, TODO: fix this
module {
  func.func @mlir_dequantizelinear_convolution_quantizelinear(%dummy : !migraphx.shaped<9x8xi8, 8x1>) -> !migraphx.shaped<1x1x7x7xui8, 49x49x7x1> {
    %arg0 = migraphx.literal (dense<[23, 28, 19, 20, 20, 31, 4, 22, 19, 30, 11, 23, 16, 27, 21, 31, 31, 3, 31, 26, 1, 27, 12, 7, 4, 7, 19, 20, 30, 4, 22, 6, 28, 22, 12, 22, 15, 2, 9, 7, 26, 10, 1, 8, 22, 13, 9, 28, 10]> : tensor<49xui8>) : <49xui8, 1>
    %arg1 = migraphx.literal (dense<0.375> : tensor<1xf32>) : <1xf32, 1>
    %arg2 = migraphx.literal (dense<21> : tensor<1xui8>) : <1xui8, 1>
    %arg3 = migraphx.literal (dense<0.1875> : tensor<1x1x7x7xf32>) : <1x1x7x7xf32, 7x7x7x1>
    %arg4 = migraphx.literal (dense<251> : tensor<1x1x7x7xui8>) : <1x1x7x7xui8, 7x7x7x1>
    %arg5 = migraphx.literal (dense<0.25> : tensor<1x1x1x1xf32>) : <1x1x1x1xf32, 1x1x1x1>

    %arg0_reshaped = migraphx.reshape %arg0 {dims = [1, 1, 7, 7]} : <49xui8, 1> -> <1x1x7x7xui8, 49x49x7x1>
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [1, 1, 7, 7]} : <1xf32, 1> -> <1x1x7x7xf32, 0x0x0x0>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 1, 7, 7]} : <1xui8, 1> -> <1x1x7x7xui8, 0x0x0x0>
    %2 = migraphx.dequantizelinear %arg0_reshaped, %0, %1 : <1x1x7x7xui8, 49x49x7x1>, <1x1x7x7xf32, 0x0x0x0>, !migraphx.shaped<1x1x7x7xui8, 0x0x0x0> -> <1x1x7x7xf32, 49x49x7x1>
    %3 = migraphx.convolution %2, %arg5 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x7x7xf32, 49x49x7x1>, <1x1x1x1xf32, 1x1x1x1> -> <1x1x7x7xf32, 49x49x7x1>
    %4 = migraphx.quantizelinear %3, %arg3, %arg4 : <1x1x7x7xf32, 49x49x7x1>, <1x1x7x7xf32, 7x7x7x1>, !migraphx.shaped<1x1x7x7xui8, 7x7x7x1> -> <1x1x7x7xui8, 49x49x7x1>
    return %4 : !migraphx.shaped<1x1x7x7xui8, 49x49x7x1>
  }
}
