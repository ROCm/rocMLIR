// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -fut test_mo -verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK-COUNT-2:  [1 1 1]
module {
    func.func @test_mo(%arg0: tensor<1x256x768xf32>, %arg1: tensor<1x768x768xf32>, %arg2: tensor<1x256x1xf32>, %arg3: tensor<1x256x768xf32>) -> (tensor<1x256x768xf32>, tensor<1x256x768xf32>) {
        %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<1x256x768xf32>, tensor<1x768x768xf32>) -> tensor<1x256x768xf32>
        %1 = "migraphx.multibroadcast"(%arg2) {out_lens = [1, 768, 768]} : (tensor<1x256x1xf32>) -> (tensor<1x256x768xf32>)
        %2 = "migraphx.add"(%0, %1) : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
        %3 = "migraphx.add"(%2, %arg3) : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
        %4 = "migraphx.relu"(%3) : (tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
        return %3, %4 : tensor<1x256x768xf32>, tensor<1x256x768xf32>
    }
}
