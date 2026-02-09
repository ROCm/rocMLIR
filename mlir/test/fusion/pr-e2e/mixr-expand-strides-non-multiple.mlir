// RUN: rocmlir-gen -fut mlir_dot_sigmoid --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_sigmoid_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s 

// Only ~41% of the results will be correct since the non-contiguous strides
// in this example is the result of concating 4x5x24 and 4x7x24 (i.e.m 4x12x24).
// This kernel is specificall dealing with making sure that the 4x5x24 elements
// are all in the correct place in the larger tensor.
// CHECK: relDiff = 0 : 480/1152 (41.666667%)

module {
  func.func @mlir_dot_sigmoid(%arg0: !migraphx.shaped<4x5x16xf16, 80x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x5x24xf16, 288x24x1> attributes {arch = "gfx1201", kernel = "mixr", num_cu = 32 : i64} {
    %0 = migraphx.dot %arg0, %arg1 : <4x5x16xf16, 80x16x1>, <4x16x24xf16, 384x24x1> -> <4x5x24xf16, 120x24x1>
    %1 = migraphx.sigmoid %0 : <4x5x24xf16, 120x24x1> -> <4x5x24xf16, 288x24x1>
    return %1 : !migraphx.shaped<4x5x24xf16, 288x24x1>
  }
}
