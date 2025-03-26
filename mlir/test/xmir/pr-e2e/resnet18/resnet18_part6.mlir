// RUN: rocmlir-gen -fut forward__part_6 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -relDiff_threshold 0.00001 -ph -fut forward__part_6_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// ALLOW_RETRIES: 2
// CHECK: [1 1 1]

module {
  func.func private @forward__part_6(%arg0: tensor<1x256x14x14xf32> {mhal.read_access}, %arg1: tensor<256x1x1xf32> {mhal.read_access}, %arg2: tensor<1x256x14x14xf32> {mhal.read_access}) -> (tensor<1x256x14x14xf32> {mhal.write_access}) {
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x256x14x14xf32>) -> tensor<1x14x14x256xf32>
    %2 = "tosa.const"() <{values = dense<-0.0425682515> : tensor<256x3x3x256xf32>}> : () -> tensor<256x3x3x256xf32>
    %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %5 = "tosa.const"() <{values = dense<-0.0823119357> : tensor<1x256x1x1xf32>}> : () -> tensor<1x256x1x1xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[1, 256, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %6 = tosa.reshape %arg1, %const_shape : (tensor<256x1x1xf32>, !tosa.shape<4>) -> tensor<1x256x1x1xf32>
    %7 = "tosa.const"() <{values = dense<0.197083265> : tensor<1x256x1x1xf32>}> : () -> tensor<1x256x1x1xf32>
    %8 = "tosa.const"() <{values = dense<-0.0162449498> : tensor<1x256x1x1xf32>}> : () -> tensor<1x256x1x1xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %9 = tosa.conv2d %1, %2, %3, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x14x14x256xf32>
    %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x14x14x256xf32>) -> tensor<1x256x14x14xf32>
    %11 = tosa.sub %10, %5 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %12 = tosa.mul %11, %6, %shift : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>, tensor<1xi8>) -> tensor<1x256x14x14xf32>
    %13 = tosa.mul %12, %7, %shift : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>, tensor<1xi8>) -> tensor<1x256x14x14xf32>
    %14 = tosa.add %13, %8 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %15 = tosa.add %14, %arg2 : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %16 = tosa.clamp %15 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    return %16 : tensor<1x256x14x14xf32>
  }
}
