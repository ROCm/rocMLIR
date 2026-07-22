// RUN: rocmlir-gen -fut mlir_quant_convolution --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -targets %arch --verify-passes | rocmlir-gen -ph -rand 1 -rand_type int -fut mlir_quant_convolution_wrapper --verifier clone -print-verify-results=always - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

module {
  func.func @mlir_quant_convolution(%arg0: !migraphx.shaped<1x4x16x16xsi8, 1024x256x16x1>, %arg1: !migraphx.shaped<16x4x3x3xsi8, 36x9x3x1>) -> !migraphx.shaped<1x16x16x16xsi32, 6144x256x16x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.quant_convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x4x16x16xsi8, 1024x256x16x1>, <16x4x3x3xsi8, 36x9x3x1> -> <1x16x16x16xsi32, 6144x256x16x1>
    return %0 : !migraphx.shaped<1x16x16x16xsi32, 6144x256x16x1>
  }
}
