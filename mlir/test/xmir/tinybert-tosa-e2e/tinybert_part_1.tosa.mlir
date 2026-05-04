// MLIR#765: TinyBERT partition 1
// RUN: rocmlir-gen -fut tinybert_part_1 -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut tinybert_part_1_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @tinybert_part_1(%arg0: tensor<2x2x128x64xf32> {mhal.read_access}, %arg1: tensor<2x2x64x128xf32> {mhal.read_access}, %arg2: tensor<1x1x1x1xf32> {mhal.read_access}, %arg3: tensor<2x1x1x128xf32> {mhal.read_access}) -> (tensor<2x2x128x128xf32> {mhal.write_access}) {
    %_shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %_s0 = "tosa.const_shape"() {values = dense<[4, 64, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %_s1 = "tosa.const_shape"() {values = dense<[4, 128, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %_s2 = "tosa.const_shape"() {values = dense<[2, 2, 128, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = "tosa.reshape"(%arg1, %_s0) : (tensor<2x2x64x128xf32>, !tosa.shape<3>) -> tensor<4x64x128xf32>
    %1 = "tosa.reshape"(%arg0, %_s1) : (tensor<2x2x128x64xf32>, !tosa.shape<3>) -> tensor<4x128x64xf32>
    %2 = "tosa.matmul"(%1, %0, %_zp, %_zp) : (tensor<4x128x64xf32>, tensor<4x64x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x128x128xf32>
    %3 = "tosa.reshape"(%2, %_s2) : (tensor<4x128x128xf32>, !tosa.shape<4>) -> tensor<2x2x128x128xf32>
    %4 = tosa.mul %3, %arg2, %_shift : (tensor<2x2x128x128xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x2x128x128xf32>
    %5 = "tosa.add"(%4, %arg3) : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tensor<2x2x128x128xf32>
    return %5 : tensor<2x2x128x128xf32>
  }
}
