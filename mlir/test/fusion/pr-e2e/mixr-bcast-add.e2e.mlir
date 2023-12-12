// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2

// CHECK:  6,     6,     6,     6
module {
  func.func @main0(%arg0: !migraphx.shaped<1x5x4x4xf32, 80x16x4x1>, %arg1: !migraphx.shaped<4x5x1x1xf32, 5x1x1x1>, %arg2: !migraphx.shaped<4xf32, 1>) -> !migraphx.shaped<1x4x4x4xf32, 64x16x4x1> attributes{kernel, arch = ""} {
    %0 = migraphx.convolution %arg0, %arg1 {padding = [0:i64, 0:i64, 0:i64, 0:i64], stride = [1:i64, 1:i64], dilation = [1:i64, 1:i64], group = 1:i64} : <1x5x4x4xf32, 80x16x4x1>, <4x5x1x1xf32, 5x1x1x1> -> <1x4x4x4xf32, 64x16x4x1>
    %1 = migraphx.broadcast %arg2 {axis = 1:i64, out_lens= [1:i64, 4:i64, 4:i64, 4:i64] } : <4xf32, 1> -> <1x4x4x4xf32, 0x1x0x0>
    %2 = migraphx.add %0, %1 {} : <1x4x4x4xf32, 64x16x4x1>, <1x4x4x4xf32, 0x1x0x0> -> <1x4x4x4xf32, 64x16x4x1>
    return %2 : !migraphx.shaped<1x4x4x4xf32, 64x16x4x1>
  }
}
