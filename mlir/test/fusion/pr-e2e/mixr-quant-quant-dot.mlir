// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver --kernel-pipeline migraphx-linalg,highlevel --host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut mlir_quantizelinear - | rocmlir-driver -arch %arch -c  | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s

// Casting/quantization semantics are the same betwaeen both kernel and host. To see if quantization is working correctly,
// we can compare the results of the linalg and the tosa pipelines instead. Doing --kernel-pipeline=migraphx-linalg --host-pipeline=migraphx-linalg
// gives the same results even if the quantization is done incorrectly.

// CHECK: [1 1 1]
module {
  func.func @mlir_quantizelinear(%dummy: !migraphx.shaped<1xf32, 1>) -> !migraphx.shaped<1x3xi32, 3x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %input = migraphx.literal (dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>) : <4xf32, 1>
    %scale = migraphx.literal (dense<5.000000e-01> : tensor<1xf32>) : <1xf32, 1>
    %bc_scale = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [4]} : <1xf32, 1> -> <4xf32, 0>
    %result = migraphx.quantizelinear %input, %bc_scale : <4xf32, 1>, <4xf32, 0> -> <4xi8, 1>
    %reshaped = migraphx.reshape %result {dims = [1, 4]} : <4xi8, 1> -> <1x4xi8, 4x1>
    %weight = migraphx.literal (dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]> : tensor<4x3xi8>) : <4x3xi8, 3x1>
    %dot_result = migraphx.quant_dot %reshaped, %weight : <1x4xi8, 4x1>, <4x3xi8, 3x1> -> <1x3xi32, 3x1>
    return %dot_result : !migraphx.shaped<1x3xi32, 3x1>
  }
}
