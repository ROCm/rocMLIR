// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void

// The CPU lowering pipeline is currently broken for backwards data convolution
// ops (lowering tosa.transpose_conv2d). As such, we do not currently have a way
// of verifying the GPU results against the CPU results. For the time being, we
// want to check that the GPU lowering pipeline can successfully lower the op
// and produce results (the first RUN command), and then for verification
// we can use rocmlir-gen to create and test a backwards data convolution op
// with the exact same shape and attributes as the one in the MIXR example below
// RUN: rocmlir-gen --operation conv_bwd_data --arch %arch -t f32 --fil_layout gkcyx --in_layout ngchw --out_layout ngkhw --batchsize 1 --groupsize 1 --in_channels 4 --out_channels 3 --in_h 11 --in_w 19 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 3 --padding_h 1 --padding_w 1 -v4r1 0 -pv | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=GEN

module {
  func.func @mlir_bwd_data_conv(%arg0: !migraphx.shaped<1x3x6x7xf32, 126x42x7x1>,
                                        %arg1: !migraphx.shaped<3x4x3x3xf32, 36x9x3x1>
                                        ) -> !migraphx.shaped<1x4x11x19xf32, 836x209x19x1> attributes {arch = "##TOKEN_ARCH##", kernel} {
    // GEN: [1 1 1]
    %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
      dilation = [1, 1],
      group = 1 : i64,
      padding = [1, 1, 1, 1],
      padding_mode = 0 : i64,
      stride = [2, 3]} : <1x3x6x7xf32, 126x42x7x1>, <3x4x3x3xf32, 36x9x3x1> -> <1x4x11x19xf32, 836x209x19x1>
    return %0 : !migraphx.shaped<1x4x11x19xf32, 836x209x19x1>
  }
}
