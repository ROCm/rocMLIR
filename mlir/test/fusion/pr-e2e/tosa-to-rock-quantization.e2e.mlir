// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK: {{.*}}rank = 4 offset = 0 sizes = [1, 4, 2, 2]{{.*}}
  func.func @mlir_quantization(%arg0: !migraphx.shaped<1x4x1x1xf32, 4x1x1x1>, %arg1: !migraphx.shaped<1x4x2x2xi8, 16x4x2x1>, %arg2: !migraphx.shaped<4x4x1x1xi8, 4x1x1x1>) -> !migraphx.shaped<1x4x2x2xi8, 16x4x2x1> attributes {arch = "", kernel = "mixr"} {
    %0 = migraphx.literal (dense<0> : tensor<1xi8>) : <1xi8, 0>
    %1 = migraphx.literal (dense<7.812500e-03> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.literal (dense<1.22070313E-4> : tensor<1xf32>) : <1xf32, 0>
    %3 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [1, 64, 56, 56]} : <1x4x1x1xf32, 4x1x1x1> -> <1x4x2x2xf32, 0x1x0x0>
    %5 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 4, 2, 2]} : <1xi8, 0> -> <1x4x2x2xi8, 0x0x0x0>
    %6 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 4, 2, 2]} : <1xf32, 0> -> <1x4x2x2xf32, 0x0x0x0>
    %7 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 4, 2, 2]} : <1xf32, 0> -> <1x4x2x2xf32, 0x0x0x0>
    %9 = migraphx.dequantizelinear %8, %7 : <1x4x2x2xi32, 16x4x2x1>, <1x4x2x2xf32, 0x0x0x0> -> <1x4x2x2xf32, 16x4x2x1>
    %10 = migraphx.add %9, %arg0 : <1x4x2x2xf32, 16x4x2x1>, <1x4x1x1xf32, 4x1x1x1> -> <1x4x2x2xf32, 16x4x2x1>
    %11 = migraphx.relu %10 : <1x4x2x2xf32, 16x4x2x1> -> <1x4x2x2xf32, 16x4x2x1>
    %12 = migraphx.quantizelinear %11, %6, %5 : <1x4x2x2xf32, 16x4x2x1>, <1x4x2x2xf32, 0x0x0x0>, !migraphx.shaped<1x4x2x2xi8, 0x0x0x0> -> <1x4x2x2xi8, 16x4x2x1>
    return %12 : !migraphx.shaped<1x4x2x2xi8, 16x4x2x1>
  }
}
