// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch,gfx908 %s | rocmlir-gen -ph -print-results -fut add - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full -targets gfx908 | xmir-runner --mlir-disable-threading --target-type cpu --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 32, 32, 64] strides = [65536, 2048, 64, 1] data =
  func.func @add(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32> {
    %9 = "tosa.add"(%arg0, %arg1)
     : (tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    return %9 : tensor<1x32x32x64xf32>
  }
}
