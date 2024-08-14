// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-driver -kernel-pipeline migraphx %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot_add --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2

// CHECK: sizes = [1024] strides = [1]
// CHECK-COUNT-1024: 17
// CLONE: [1 1 1]
module {
  func.func @mlir_dot_add(%arg0: !migraphx.shaped<4x8x32xf16, 0x32x1>,
                          %arg1: !migraphx.shaped<4x8x16xf16, 128x16x1>,
                          %arg2: !migraphx.shaped<16x32xf16, 32x1>) -> !migraphx.shaped<4x8x32xf16, 256x32x1> attributes {arch = "", kernel} {
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [4, 16, 32]} : <16x32xf16, 32x1> -> <4x16x32xf16, 0x32x1>
    %1 = migraphx.dot %arg1, %0 : <4x8x16xf16, 128x16x1>, <4x16x32xf16, 0x32x1> -> <4x8x32xf16, 256x32x1>
    %2 = migraphx.add %1, %arg0 : <4x8x32xf16, 256x32x1>, <4x8x32xf16, 0x32x1> -> <4x8x32xf16, 256x32x1>
    return %2 : !migraphx.shaped<4x8x32xf16, 256x32x1>
  }
}


