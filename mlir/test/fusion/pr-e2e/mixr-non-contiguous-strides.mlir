// RUN: rocmlir-gen -fut mlir_dot_sigmoid --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_sigmoid_wrapper --verifier clone -print-verify-results=always - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s 

// Non-contiguous strides: validates expand-strides / padded layout. With
// correct GPU host staging, CPU and GPU references match on the full tensor.
// CHECK: relDiff = 0 : 4608/4608 (100.000000%)

module {
  func.func @mlir_dot_sigmoid(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x24x1> attributes {kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
    %1 = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x24x1>
    return %1 : !migraphx.shaped<4x24x24xf16, 1152x24x1>
  }
}

