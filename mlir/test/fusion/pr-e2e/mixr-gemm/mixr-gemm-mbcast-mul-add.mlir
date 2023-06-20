// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base

func.func @mlir_dot(%arg0: tensor<1x1x1x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x12x384x64xf32>, %arg3: tensor<1x12x384x64xf32>) -> tensor<1x12x384x384xf32> attributes{kernel, arch = ""} {
    %0 = migraphx.multibroadcast(%arg1) {out_dyn_dims = [], out_lens = [1, 12, 384, 384]} : (tensor<1x1x1x1xf32>) -> tensor<1x12x384x384xf32>
    %1 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [1, 12, 384, 384]} : (tensor<1x1x1x1xf32>) -> tensor<1x12x384x384xf32>
    %2 = migraphx.transpose(%arg3) {permutation = [0, 1, 3, 2]} : (tensor<1x12x384x64xf32>) -> tensor<1x12x64x384xf32>
    %3 = migraphx.dot(%arg2, %2) : (tensor<1x12x384x64xf32>, tensor<1x12x64x384xf32>) -> tensor<1x12x384x384xf32>
    %4 = migraphx.mul(%3, %1) : (tensor<1x12x384x384xf32>, tensor<1x12x384x384xf32>) -> tensor<1x12x384x384xf32>
    %5 = migraphx.add(%4, %0) : (tensor<1x12x384x384xf32>, tensor<1x12x384x384xf32>) -> tensor<1x12x384x384xf32>
    return %5 : tensor<1x12x384x384xf32>
  }
}
