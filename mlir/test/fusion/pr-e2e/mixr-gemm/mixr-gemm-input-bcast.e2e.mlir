// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK:  {{.*}}[4, 4, 4, 4],
  // CHECK-NEXT:   [4, 4, 4, 4]{{.*}}
  func.func @mlir_dot(%arg0: !migraphx.shaped<1x2x4xf32, 8x4x1>, %arg1: !migraphx.shaped<1x2x3xf32, 1x0x0>, %arg2: !migraphx.shaped<1x3x4xf32, 12x4x1>) -> !migraphx.shaped<1x2x4xf32, 8x4x1> attributes{kernel, arch = ""} {
    %1 = migraphx.dot %arg1, %arg2 : <1x2x3xf32, 1x0x0>, <1x3x4xf32, 12x4x1> -> <1x2x4xf32, 8x4x1>
    %2 = migraphx.add %1, %arg0 : <1x2x4xf32, 8x4x1>, <1x2x4xf32, 8x4x1> -> <1x2x4xf32, 8x4x1>
    return %2 : !migraphx.shaped<1x2x4xf32, 8x4x1>
  }
}
