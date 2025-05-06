// RUN: rocmlir-gen -fut forward__part_0 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -fut forward__part_0_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// ALLOW_RETRIES: 2 
// CHECK: [1 1 1]

module {
  func.func private @forward__part_0(%arg0: tensor<1x1x1x512xf32> {mhal.read_access}) -> (tensor<1x1000xf32> {mhal.write_access}) {
    %const_shape = "tosa.const_shape"() { values = dense<[1, 1, 512]> : tensor<3xindex> } : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %const_shape : (tensor<1x1x1x512xf32>, !tosa.shape<3>) -> tensor<1x1x512xf32>
    %1 = "tosa.const"() <{values = dense<-0.0184740368> : tensor<1x512x1000xf32>}> : () -> tensor<1x512x1000xf32>
    %2 = "tosa.const"() <{values = dense<-0.00263410225> : tensor<1x1000xf32>}> : () -> tensor<1x1000xf32>
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.matmul %0, %1, %a_zp, %b_zp : (tensor<1x1x512xf32>, tensor<1x512x1000xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1000xf32>
    %const_shape2 = "tosa.const_shape"() { values = dense<[1, 1000]> : tensor<2xindex> } : () -> !tosa.shape<2>
    %4 = tosa.reshape %3, %const_shape2 : (tensor<1x1x1000xf32>, !tosa.shape<2>) -> tensor<1x1000xf32>
    %5 = tosa.add %4, %2 : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %5 : tensor<1x1000xf32>
  }
}
