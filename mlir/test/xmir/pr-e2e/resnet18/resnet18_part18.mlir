// RUN: rocmlir-gen -fut forward__part_18 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -relDiff_threshold 0.00001 -ph -fut forward__part_18_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

module {
  func.func private @forward__part_18(%arg0: tensor<1x64x56x56xf32> {mhal.read_access}, %arg1: tensor<1x64x1x1xf32> {mhal.read_access}, %arg2: tensor<64x1x1xf32> {mhal.read_access}, %arg3: tensor<1x64x1x1xf32> {mhal.read_access}, %arg4: tensor<1x64x1x1xf32> {mhal.read_access}, %arg5: tensor<1x56x56x64xf32> {mhal.read_access}) -> (tensor<1x64x56x56xf32> {mhal.write_access}) {
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
    %2 = "tosa.const"() <{values = dense<0.0259467717> : tensor<64x3x3x64xf32>}> : () -> tensor<64x3x3x64xf32>
    %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[1, 64, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %5 = tosa.reshape %arg2, %const_shape : (tensor<64x1x1xf32>, !tosa.shape<4>) -> tensor<1x64x1x1xf32>
    %6 = tosa.transpose %arg5 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x56x56x64xf32>) -> tensor<1x64x56x56xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %7 = tosa.conv2d %1, %2, %3, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x56x56x64xf32>
    %8 = tosa.transpose %7 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x56x56x64xf32>) -> tensor<1x64x56x56xf32>
    %9 = tosa.sub %8, %arg1 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %10 = tosa.mul %9, %5, %shift : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>, tensor<1xi8>) -> tensor<1x64x56x56xf32>
    %11 = tosa.mul %10, %arg3, %shift : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>, tensor<1xi8>) -> tensor<1x64x56x56xf32>
    %12 = tosa.add %11, %arg4 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %13 = tosa.add %12, %6 : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %14 = tosa.clamp %13 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return %14 : tensor<1x64x56x56xf32>
  }
}
