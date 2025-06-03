// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_convert_convolution %s | rocmlir-driver  -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut mlir_convert_convolution_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]

module {
  func.func @mlir_convert_convolution(%arg0: !migraphx.shaped<1x1x128x128x128xf32, 2097152x2097152x16384x128x1>, %arg1: !migraphx.shaped<32x1x3x3x3xf16, 27x27x9x3x1>) -> !migraphx.shaped<1x32x128x128x128xf16, 67108864x2097152x16384x128x1> {
    %0 = migraphx.convert %arg0 {target_type = 1 : i64} : <1x1x128x128x128xf32, 2097152x2097152x16384x128x1> to <1x1x128x128x128xf16, 2097152x2097152x16384x128x1>
    %1 = migraphx.convolution %0, %arg1 {dilation = [1, 1, 1], group = 1 : i64, padding = [1, 1, 1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1, 1]} : <1x1x128x128x128xf16, 2097152x2097152x16384x128x1>, <32x1x3x3x3xf16, 27x27x9x3x1> -> <1x32x128x128x128xf16, 67108864x2097152x16384x128x1>
    return %1 : !migraphx.shaped<1x32x128x128x128xf16, 67108864x2097152x16384x128x1>
  }
}
