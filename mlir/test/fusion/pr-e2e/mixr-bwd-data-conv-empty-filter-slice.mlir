// RUN: rocmlir-gen -fut mlir_bwd_data_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_bwd_data_conv_wrapper --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// We also want to check the original rocmlir-gen command that initially hit
// this issue is fixed. The MIGraphX -> Tosa -> Linalg CPU lowering currently does not
// support large group sizes (> 1), so we need to use the original rocmlir-gen command
// to check that the issue is fixed with g=4.
// RUN: rocmlir-gen --operation conv_bwd_data -t f16 --arch %arch -v4r1 0 --kernel_id 0 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 1 --in_channels 64 --in_h 32 --in_w 14 --out_channels 256 --fil_h 2 --fil_w 1 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 3 --padding_h 0 --padding_w 3 --groupsize 4 --perf_config= -pv | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=ROCKGEN

// ROCKGEN: [1 1 1]

module {
  // CHECK: [1 1 1]
  func.func @mlir_bwd_data_conv(%grad_out: !migraphx.shaped<1x16x16x7xf16, 1792x112x7x1>, %weights: !migraphx.shaped<16x16x2x1xf16, 32x2x1x1>) -> !migraphx.shaped<1x16x32x19xf16, 9728x608x19x1> attributes {rock.kernel} {
    %res = migraphx.backwards_data_convolution %grad_out, %weights {
      dilation = [1, 2],
      group = 1 : i64,
      padding = [0, 0, 0, 0],
      padding_mode = 0 : i64,
      stride = [2, 3]
    } : <1x16x16x7xf16, 1792x112x7x1>, <16x16x2x1xf16, 32x2x1x1> -> <1x16x32x19xf16, 9728x608x19x1>
    return %res : !migraphx.shaped<1x16x32x19xf16, 9728x608x19x1>
  }
}
