// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 4 -rand_type float -fut forward__part_0 --verifier clone - | rocmlir-driver -host-pipeline xmodel,runner -kernel-pipeline full -targets %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  func.func @forward__part_0(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<64x7x7x3xf32>, %arg2: tensor<1x64x1x1xf32>) -> (tensor<1x64x112x112xf32>) attributes {kernel} {
    %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>
    %1 = "tosa.conv2d"(%arg0, %arg1, %0) {dilation = [1, 1], pad = [3, 3, 3, 3], stride = [2, 2]} : (tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
    %2 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %3 = "tosa.transpose"(%1, %2) : (tensor<1x112x112x64xf32>, tensor<4xi32>) -> tensor<1x64x112x112xf32>
    %4 = "tosa.sub"(%3, %arg2) : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    return %4 : tensor<1x64x112x112xf32>
  }
}
