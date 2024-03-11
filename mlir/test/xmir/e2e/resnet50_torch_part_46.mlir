// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut resnet50_part_46 --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: RMS = {{.*}}e-08
// CHECK: [1 1 0]
module {
  func.func @resnet50_part_46(%arg0: tensor<2x1024x13x13xf32> {func.read_access}, %arg1: tensor<2048x1024x1x1xf32> {func.read_access}, %arg2: tensor<1x2048x1x1xf32> {func.read_access}, %arg3: tensor<1x2048x1x1xf32> {func.read_access}, %arg4: tensor<1x2048x1x1xf32> {func.read_access}, %arg5: tensor<1x2048x1x1xf32> {func.read_access}) -> (tensor<2x2048x7x7xf32> {func.write_access}) {
    %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<2048xf32>} : () -> tensor<2048xf32>
    %1 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %2 = "tosa.transpose"(%arg1, %1) : (tensor<2048x1024x1x1xf32>, tensor<4xi32>) -> tensor<2048x1x1x1024xf32>
    %3 = "tosa.transpose"(%arg0, %1) : (tensor<2x1024x13x13xf32>, tensor<4xi32>) -> tensor<2x13x13x1024xf32>
    %4 = "tosa.conv2d"(%3, %2, %0) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<2x13x13x1024xf32>, tensor<2048x1x1x1024xf32>, tensor<2048xf32>) -> tensor<2x7x7x2048xf32>
    %5 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %6 = "tosa.transpose"(%4, %5) : (tensor<2x7x7x2048xf32>, tensor<4xi32>) -> tensor<2x2048x7x7xf32>
    %7 = "tosa.sub"(%6, %arg2) : (tensor<2x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<2x2048x7x7xf32>
    %8 = "tosa.mul"(%7, %arg3) {shift = 0 : i32} : (tensor<2x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<2x2048x7x7xf32>
    %9 = "tosa.mul"(%8, %arg4) {shift = 0 : i32} : (tensor<2x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<2x2048x7x7xf32>
    %10 = "tosa.add"(%9, %arg5) : (tensor<2x2048x7x7xf32>, tensor<1x2048x1x1xf32>) -> tensor<2x2048x7x7xf32>
    return %10 : tensor<2x2048x7x7xf32>
  }
}
