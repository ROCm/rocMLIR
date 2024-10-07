// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_convolution_add %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -rand 0 -rand_type float -fut mlir_convolution_add_wrapper --perf_config="v2:32,128,8,16,16,4,1,1,1" - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full --arch %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]

func.func @mlir_convolution_add(%arg0: !migraphx.shaped<8x16x50x50x100xf32, 0x1x0x0x0>, %arg1: !migraphx.shaped<8x3x50x50x100xf32, 750000x250000x5000x100x1>, %arg2: !migraphx.shaped<16x3x5x5x5xf32, 375x125x25x5x1>) -> !migraphx.shaped<8x16x50x50x100xf32, 4000000x250000x5000x100x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr", num_cu = 110 : i64} {
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [2, 2, 2], group = 1 : i64, padding = [4, 4, 4, 4, 4, 4], padding_mode = 0 : i64, stride = [1, 1, 1]} : <8x3x50x50x100xf32, 750000x250000x5000x100x1>, <16x3x5x5x5xf32, 375x125x25x5x1> -> <8x16x50x50x100xf32, 4000000x250000x5000x100x1>
  %1 = migraphx.add %0, %arg0 : <8x16x50x50x100xf32, 4000000x250000x5000x100x1>, <8x16x50x50x100xf32, 0x1x0x0x0> -> <8x16x50x50x100xf32, 4000000x250000x5000x100x1>
  return %1 : !migraphx.shaped<8x16x50x50x100xf32, 4000000x250000x5000x100x1>
}
