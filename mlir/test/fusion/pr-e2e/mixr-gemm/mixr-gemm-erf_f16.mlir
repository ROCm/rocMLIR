// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand 3 - | rocmlir-driver -arch %arch -c  | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
module {
  // CHECK:  [-1, 1, 1, -1, 1, -1, -1, 1, -1, {{.*}}, -1, -1, 0, -1, -1]
  // NOTE: the missing number is -1.07288e-06 in navi instead of 0
  func.func @dot_add(%arg0: !migraphx.shaped<1x5x4xf16, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf16, 12x3x1>) -> !migraphx.shaped<1x5x3xf16, 15x3x1> attributes{kernel, arch = "##TOKEN_ARCH##"} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf16, 20x4x1>, <1x4x3xf16, 12x3x1> -> <1x5x3xf16, 15x3x1>
    %2 = migraphx.erf %0 : <1x5x3xf16, 15x3x1> -> <1x5x3xf16, 15x3x1>
    return %2 : !migraphx.shaped<1x5x3xf16, 15x3x1>
  }
}
