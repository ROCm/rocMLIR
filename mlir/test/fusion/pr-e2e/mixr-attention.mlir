// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-driver -host-pipeline=partition,highlevel -targets %arch | rocmlir-gen -ph -fut mlir_attention --verifier clone - | rocmlir-driver -c -arch %arch | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
// Note the fake wrapper function to make it look like this has already been partitioned,
// since we want "partition" for the xmodel packaging but don't actually want to use
// the partition logic.
// This is an awkward hack that should be fixed Later (tm)
module {
  func.func @mlir_attention_real(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {kernel = "mixr", num_cu = 48 : i64} {
    %0 = migraphx.dot(%arg0, %arg1): (tensor<1x384x64xf32>, tensor<1x64x384xf32>) -> tensor<1x384x384xf32>
    %1 = migraphx.softmax(%0){axis = 2 : i64} : tensor<1x384x384xf32> -> tensor<1x384x384xf32>
    %2 = migraphx.dot(%1, %arg2): (tensor<1x384x384xf32>, tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
    return %2 : tensor<1x384x64xf32>
  }

  func.func @mlir_attention(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> {
    %ret = call @mlir_attention_real(%arg0, %arg1, %arg2) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>) -> (tensor<1x384x64xf32>)
    return %ret : tensor<1x384x64xf32>
  }
}