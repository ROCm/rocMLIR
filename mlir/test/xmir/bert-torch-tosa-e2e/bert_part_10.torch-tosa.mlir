// RUN: rocmlir-gen -fut bert_part_10 -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -targets %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut bert_part_10_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @bert_part_10(%arg0: tensor<1x12x384xf32> {mhal.read_access}, %arg1: tensor<384x384xf32> {mhal.read_access}, %arg2: tensor<1x1x384xf32> {mhal.read_access}) -> (tensor<1x12x12x32xf32> {mhal.write_access}) {
      %const_shape = "tosa.const_shape"() { values = dense<[1, 384, 384]> : tensor<3xindex> } : () -> !tosa.shape<3>
      %0 = "tosa.reshape"(%arg1, %const_shape) : (tensor<384x384xf32>, !tosa.shape<3>) -> tensor<1x384x384xf32>
      %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
      %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
      %1 = "tosa.matmul"(%arg0, %0, %a_zp, %b_zp) : (tensor<1x12x384xf32>, tensor<1x384x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x12x384xf32>
      %2 = "tosa.add"(%1, %arg2) : (tensor<1x12x384xf32>, tensor<1x1x384xf32>) -> tensor<1x12x384xf32>
      %const_shape2 = "tosa.const_shape"() { values = dense<[1, 12, 12, 32]> : tensor<4xindex> } : () -> !tosa.shape<4>
      %3 = "tosa.reshape"(%2, %const_shape2) : (tensor<1x12x384xf32>, !tosa.shape<4>) -> tensor<1x12x12x32xf32>
      return %3 : tensor<1x12x12x32xf32>
    }
}
