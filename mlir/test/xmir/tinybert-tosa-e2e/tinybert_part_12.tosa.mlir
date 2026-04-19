// MLIR#765: TinyBERT partition 12
// RUN: rocmlir-gen -fut tinybert_part_12 -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut tinybert_part_12_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @tinybert_part_12(%arg0: tensor<2x128x128xf32> {mhal.read_access}, %arg1: tensor<1x128x128xf32> {mhal.read_access}, %arg2: tensor<1x1x128xf32> {mhal.read_access}) -> (tensor<2x128x128xf32> {mhal.write_access}, tensor<2x128x128xf32> {mhal.write_access}) {
    %_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %_s0 = "tosa.const_shape"() {values = dense<[1, 256, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %_s1 = "tosa.const_shape"() {values = dense<[2, 128, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = "tosa.reshape"(%arg0, %_s0) : (tensor<2x128x128xf32>, !tosa.shape<3>) -> tensor<1x256x128xf32>
    %1 = "tosa.matmul"(%0, %arg1, %_zp, %_zp) : (tensor<1x256x128xf32>, tensor<1x128x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x128xf32>
    %2 = "tosa.reshape"(%1, %_s1) : (tensor<1x256x128xf32>, !tosa.shape<3>) -> tensor<2x128x128xf32>
    %3 = "tosa.add"(%2, %arg2) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %5 = "tosa.sub"(%3, %4) : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    return %3, %5 : tensor<2x128x128xf32>, tensor<2x128x128xf32>
  }
}
