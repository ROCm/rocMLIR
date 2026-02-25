// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=MIXR
// RUN: rocmlir-gen -fut mlir_bwd_data_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx-linalg,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_bwd_data_conv_wrapper --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=LINALG

// The CPU lowering pipeline is currently broken for backwards data convolution
// ops (lowering tosa.transpose_conv2d). As such, we do not currently have a way
// of verifying the GPU results against the CPU results. For the time being, we
// want to check that the GPU lowering pipeline can successfully lower the op
// and produce results (the first RUN command), and then for verification
// we can use rocmlir-gen to create and test a backwards data convolution op
// with the exact same shape and attributes as the one in the MIXR example below
// RUN: rocmlir-gen --operation conv_bwd_data --arch %arch -t f32 --fil_layout gkcyx --in_layout ngchw --out_layout ngkhw --batchsize 1 --groupsize 1 --in_channels 1 --out_channels 1 --in_h 5 --in_w 5 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -v4r1 0 -pv | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=GEN

// TODO: We are actually generating a 2D conv with rocmlir-gen here since it does not support generating 3D
//       This should be a 3D conv once rocmlir-gen supports it.
module {
  // LINALG: [1 1 1]
  // MIXR: [1,  2,  3,  2,  1,  2,  4,  6,  4,  2,  3,  6,  9,  6,  3,  2,  4,  6,  4,  2,  1,  2,  3,  2,  1]
  // GEN: [1 1 1]
  func.func @mlir_bwd_data_conv(
      %arg0: !migraphx.shaped<1x1x1x3x3xf32, 9x9x9x3x1>,
      %arg1: !migraphx.shaped<1x1x1x3x3xf32, 9x9x9x3x1>
  ) -> !migraphx.shaped<1x1x1x5x5xf32, 25x25x25x5x1> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %0 = migraphx.backwards_data_convolution %arg1, %arg0 {
      dilation = [1, 1, 1],
      group = 1 : i64,
      padding = [0, 0, 0, 0, 0, 0],
      padding_mode = 0 : i64,
      stride = [1, 1, 1]
    } : <1x1x1x3x3xf32, 9x9x9x3x1>, <1x1x1x3x3xf32, 9x9x9x3x1> -> <1x1x1x5x5xf32, 25x25x25x5x1>
    return %0 : !migraphx.shaped<1x1x1x5x5xf32, 25x25x25x5x1>
  }
}
