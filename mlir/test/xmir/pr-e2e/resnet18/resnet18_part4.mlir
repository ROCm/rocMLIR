// RUN: rocmlir-gen -fut forward__part_4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -fut forward__part_4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

module {
  func.func private @forward__part_4(%arg0: tensor<1x512x7x7xf32> {mhal.read_access}, %arg1: tensor<512x1x1xf32> {mhal.read_access}) -> (tensor<1x512x7x7xf32> {mhal.write_access}) {
    %0 = "tosa.const"() <{values = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x7x7xf32>) -> tensor<1x7x7x512xf32>
    %2 = "tosa.const"() <{values = dense<1.62181284E-4> : tensor<512x3x3x512xf32>}> : () -> tensor<512x3x3x512xf32>
    %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %5 = "tosa.const"() <{values = dense<-0.232302353> : tensor<1x512x1x1xf32>}> : () -> tensor<1x512x1x1xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[1, 512, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %6 = tosa.reshape %arg1, %const_shape : (tensor<512x1x1xf32>, !tosa.shape<4>) -> tensor<1x512x1x1xf32>
    %7 = "tosa.const"() <{values = dense<0.447422773> : tensor<1x512x1x1xf32>}> : () -> tensor<1x512x1x1xf32>
    %8 = "tosa.const"() <{values = dense<-0.175875843> : tensor<1x512x1x1xf32>}> : () -> tensor<1x512x1x1xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %9 = tosa.conv2d %1, %2, %3, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x512xf32>, tensor<512x3x3x512xf32>, tensor<512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x512xf32>
    %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x7x7x512xf32>) -> tensor<1x512x7x7xf32>
    %11 = tosa.sub %10, %5 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %12 = tosa.mul %11, %6, %shift : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>, tensor<1xi8>) -> tensor<1x512x7x7xf32>
    %13 = tosa.mul %12, %7, %shift : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>, tensor<1xi8>) -> tensor<1x512x7x7xf32>
    %14 = tosa.add %13, %8 : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x7x7xf32>
    return %14 : tensor<1x512x7x7xf32>
  }
}
