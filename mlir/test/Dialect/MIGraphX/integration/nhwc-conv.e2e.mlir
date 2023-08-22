// RUN: rocmlir-gen --clone-harness -arch %arch -fut convNHWC %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut convNHWC_wrapper - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full --arch %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

func.func @convNHWC(%in: !migraphx.shaped<1x4x5x5xf32, 100x1x20x4>, %fil: !migraphx.shaped<7x4x3x3xf32, 36x1x12x4>) -> !migraphx.shaped<1x7x3x3xf32, 63x1x21x7> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : !migraphx.shaped<1x4x5x5xf32, 100x1x20x4>, !migraphx.shaped<7x4x3x3xf32, 36x1x12x4> -> !migraphx.shaped<1x7x3x3xf32, 63x1x21x7>
  func.return %out : !migraphx.shaped<1x7x3x3xf32, 63x1x21x7>
}
