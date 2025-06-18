// RUN: rocmlir-gen -fut test_mo --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -fut test_mo_wrapper -rand 1 -rand_type float --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext -entry-point-result=void | FileCheck %s

// CHECK-COUNT-2:  [1 1 1]
module {
  func.func @test_mo(%arg0: tensor<1x256x768xf32>, %arg1: tensor<1x768x768xf32>, %arg2: tensor<1x256x1xf32>, %arg3: tensor<1x256x768xf32>, %arg4: tensor<1x256x768xf32>) -> (tensor<1x256x768xf32>, tensor<1x256x768xf32>, tensor<1x256x768xf32>) {

    %const_shape = "tosa.const_shape"() { values = dense<[1, 256, 768]> : tensor<3xindex> } : () -> !tosa.shape<3>
    %0 = "tosa.reshape"(%arg0, %const_shape) : (tensor<1x256x768xf32>, !tosa.shape<3>) -> tensor<1x256x768xf32>
    %const_shape2 = "tosa.const_shape"() { values = dense<[1, 768, 768]> : tensor<3xindex> } : () -> !tosa.shape<3>
    %1 = "tosa.reshape"(%arg1, %const_shape2) : (tensor<1x768x768xf32>, !tosa.shape<3>) -> tensor<1x768x768xf32>
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "tosa.matmul"(%0, %1, %a_zp, %b_zp) : (tensor<1x256x768xf32>, tensor<1x768x768xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x768xf32>
    %const_shape3 = "tosa.const_shape"() { values = dense<[1, 256, 768]> : tensor<3xindex> } : () -> !tosa.shape<3>
    %3 = "tosa.reshape"(%2, %const_shape3) : (tensor<1x256x768xf32>, !tosa.shape<3>) -> tensor<1x256x768xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<1x256x768xf32>, tensor<1x256x1xf32>) -> tensor<1x256x768xf32>
    %5 = "tosa.add"(%4, %arg3) : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %6 = "tosa.clamp"(%5) {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %7 = "tosa.sub"(%6, %arg4) : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    return %5, %6, %7 : tensor<1x256x768xf32>, tensor<1x256x768xf32>, tensor<1x256x768xf32>
  }
}
