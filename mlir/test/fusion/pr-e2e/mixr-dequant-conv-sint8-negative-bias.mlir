// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut mlir_dequantizelinear_convolution_quantizelinear --arch %arch --clone-harness - | rocmlir-driver -host-pipeline=highlevel | rocmlir-gen -print-results -ph -fut mlir_dequantizelinear_convolution_quantizelinear_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [49] strides = [1] data =
// CHECK-NEXT: [1.3125, 0.84375, 1.6875, 1.59375, 1.59375, 0.5625, 3.09375, 1.40625, 1.6875, 0.65625, 2.4375, 1.3125, 1.96875, 0.9375, 1.5, 0.5625, 0.5625, 3.1875, 0.5625, 1.03125, 3.375, 0.9375, 2.34375, 2.8125, 3.09375, 2.8125, 1.6875, 1.59375, 0.65625, 3.09375, 1.40625, 2.90625, 0.84375, 1.40625, 2.34375, 1.40625, 2.0625, 3.28125, 2.625, 2.8125, 1.03125, 2.53125, 3.375, 2.71875, 1.40625, 2.25, 2.625, 0.84375, 2.53125]
// COM: tests fail is they have no arguments, that's why we have %dummy
module {
  func.func @mlir_dequantizelinear_convolution_quantizelinear(%dummy : !migraphx.shaped<9x8xi8, 8x1>) -> !migraphx.shaped<1x1x7x7xf32, 49x49x7x1> {
    %arg0 = migraphx.literal (dense<[-7, -12, -3, -4, -4, -15, 12, -6, -3, -14, 5, -7, 0, -11, -5, -15, -15, 13, -15, -10, 15, -11, 4, 9, 12, 9, -3, -4, -14, 12, -6, 10, -12, -6, 4, -6, 1, 14, 7, 9, -10, 6, 15, 8, -6, 3, 7, -12, 6]> : tensor<49xsi8>) : <49xsi8, 1>
    %arg1 = migraphx.literal (dense<0.375> : tensor<1xf32>) : <1xf32, 1>
    %arg2 = migraphx.literal (dense<-21> : tensor<1xsi8>) : <1xsi8, 1>
    %arg3 = migraphx.literal (dense<0.25> : tensor<1x1x1x1xf32>) : <1x1x1x1xf32, 1x1x1x1>

    %arg0_reshaped = migraphx.reshape %arg0 {dims = [1, 1, 7, 7]} : <49xsi8, 1> -> <1x1x7x7xsi8, 49x49x7x1>
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [1, 1, 7, 7]} : <1xf32, 1> -> <1x1x7x7xf32, 0x0x0x0>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 1, 7, 7]} : <1xsi8, 1> -> <1x1x7x7xsi8, 0x0x0x0>
    %2 = migraphx.dequantizelinear %arg0_reshaped, %0, %1 : <1x1x7x7xsi8, 49x49x7x1>, <1x1x7x7xf32, 0x0x0x0>, !migraphx.shaped<1x1x7x7xsi8, 0x0x0x0> -> <1x1x7x7xf32, 49x49x7x1>
    %3 = migraphx.convolution %2, %arg3 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x7x7xf32, 49x49x7x1>, <1x1x1x1xf32, 1x1x1x1> -> <1x1x7x7xf32, 49x49x7x1>
    return %3 : !migraphx.shaped<1x1x7x7xf32, 49x49x7x1>
  }
}
