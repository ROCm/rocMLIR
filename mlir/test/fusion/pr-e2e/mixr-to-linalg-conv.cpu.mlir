// RUN: rocmlir-gen -fut conv -arch %arch --clone-harness %s | rocmlir-driver --host-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -print-results -rand_type float -rand_min 0 -rand_max 0 -fut conv_wrapper --verifier clone -  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=GOLD
// RUN: rocmlir-gen -fut conv -arch %arch --clone-harness %s | rocmlir-driver --host-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type=float -fut conv_wrapper --verifier clone -  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=BOTH
// RUN: rocmlir-gen -fut conv -arch %arch --clone-harness %s | rocmlir-driver --host-pipeline=migraphx,highlevel --kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type=float -fut conv_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=BOTH
// RUN: rocmlir-gen -fut conv -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx-linalg,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut conv_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=FINAL

/// README - There are essentially three tests (BOTH, GOLD, and FINAL). 
/// BOTH checks if the tosa pipeline gives the same value (given the 
/// same seed) as the linalg pipeline. They will pass if both of them 
/// returns the same value. GOLD checks if the output for the linalg pipeline 
/// matches an equivalent pytorch implementation. FINAL verifies if the linalg 
/// pipeline can be converted to rock

/// Gold value computed as the following:
///
/// # These patterns are from the rocmlir-gen
/// pattern = torch.tensor([0.5, -1.0, 0.75], dtype=torch.float32) 
/// flat_x = torch.tensor([pattern[i % 3].item() for i in range(750)], dtype=torch.float32)
/// x_nchwd = flat_x.reshape(2, 3, 5, 5, 5)  # N, C, H, W, D
/// x = x_nchwd.permute(0, 1, 4, 2, 3)  # -> (N, C, D, H, W)
/// 
/// flat_w = torch.tensor([pattern[i % 3].item() for i in range(96)], dtype=torch.float32)
/// w_fchwd = flat_w.reshape(4, 3, 2, 2, 2)  # F, C, H, W, D
/// weight = w_fchwd.permute(0, 1, 4, 2, 3)  # -> (F, C, kD, kH, kW)
/// out = torch.nn.functional.conv3d(
///     x, weight,
///     stride=(2, 2, 2),
///     dilation=(2, 2, 2),
///     padding=0,
///     groups=1,
/// )
/// out_nfhwd = out.permute(0, 1, 3, 4, 2)
/// flat_out = out_nfhwd.reshape(-1)
/// 
/// print("Full 64:", flat_out.tolist()) 
/// Outputs: 
/// Full 64: [1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625]


module{
  // FINAL: [1 1 1]
  // BOTH: [6.09101, 7.06269, 5.96599, 7.63177, 5.83172, 5.96893, 5.16868, 6.0204, 6.80761, 6.78844, 5.75672, 7.33505, 5.417{{.*}}, 6.04153, 5.14715, 6.728{{.*}}, 7.30343, 7.90745, 6.73162, 8.21738, 5.65554, 7.37453, 6.6329, 6.6093, 5.2816, 6.17693, 5.19904, 6.38292, 4.55713, 4.62921, 4.72307, 5.47466, 4.551, 6.15787, 4.97358, 5.89798, 5.10684, 6.01542, 5.18933, 5.58596, 5.22862, 7.13881, 4.88134, 5.56315, 5.52007, 6.27824, 4.93779, 5.71044, 6.27934, 7.51976, 5.23159, 7.17014, 6.74235, 5.59631, 5.33666, 6.20902, 4.95302, 5.26817, 4.50571, 5.17464, 4.49137, 4.80133, 3.39298, 4.92709]
  // GOLD: [1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625, -1.625, 1.0625, 1.0625]
  func.func @conv(%arg1: !migraphx.shaped<2x3x5x5x5xf32, 375x125x25x5x1>, %arg2: !migraphx.shaped<4x3x2x2x2xf32, 24x8x4x2x1>) -> !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1> { 
    %0 = migraphx.convolution %arg1, %arg2 {dilation = [2, 2, 2], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} : <2x3x5x5x5xf32, 375x125x25x5x1>, <4x3x2x2x2xf32, 24x8x4x2x1> -> <2x4x2x2x2xf32, 32x8x4x2x1>
    return %0 : !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>
  }
}
