// RUN: rocmlir-gen -fut bert_part_19 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -fut bert_part_19_wrapper -rand 1 -rand_type float --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext -entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
    func.func @bert_part_19(%arg0: tensor<1x12x12x32xf32> {mhal.read_access}, %arg1: tensor<1x12x32x12xf32> {mhal.read_access}, %arg2: tensor<1x1x1x1xf32> {mhal.read_access}, %arg3: tensor<1x1x1x12xf32> {mhal.read_access}) -> (tensor<1x12x12x12xf32> {mhal.write_access}) {

      %const_shape = "tosa.const_shape"() { value = dense<[12, 32, 12]> : tensor<3xindex> } : () -> !tosa.shape<3>
      %0 = "tosa.reshape"(%arg1, %const_shape) : (tensor<1x12x32x12xf32>, !tosa.shape<3>) -> tensor<12x32x12xf32>
      %const_shape2 = "tosa.const_shape"() { value = dense<[12, 12, 32]> : tensor<3xindex> } : () -> !tosa.shape<3>
      %1 = "tosa.reshape"(%arg0, %const_shape2) : (tensor<1x12x12x32xf32>, !tosa.shape<3>) -> tensor<12x12x32xf32>
      %2 = "tosa.matmul"(%1, %0) : (tensor<12x12x32xf32>, tensor<12x32x12xf32>) -> tensor<12x12x12xf32>
      %const_shape3 = "tosa.const_shape"() { value = dense<[1, 12, 12, 12]> : tensor<4xindex> } : () -> !tosa.shape<4>
      %3 = "tosa.reshape"(%2, %const_shape3) : (tensor<12x12x12xf32>, !tosa.shape<4>) -> tensor<1x12x12x12xf32>
      %shift = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
      %4 = "tosa.mul"(%3, %arg2, %shift) : (tensor<1x12x12x12xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x12x12x12xf32>
      %5 = "tosa.add"(%4, %arg3) : (tensor<1x12x12x12xf32>, tensor<1x1x1x12xf32>) -> tensor<1x12x12x12xf32>
      return %5 : tensor<1x12x12x12xf32>
    }
}
