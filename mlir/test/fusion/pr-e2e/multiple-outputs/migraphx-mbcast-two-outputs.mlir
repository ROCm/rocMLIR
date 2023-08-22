// RUN: rocmlir-driver -kernel-pipeline migraphx %s | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -ph -print-results -fut test_mo -verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK-COUNT-2:  [1 1 1]
module {
    func.func @test_mo(%arg0: !migraphx.shaped<1x256x768xf32, 196608x768x1>, %arg1: !migraphx.shaped<1x768x768xf32, 589824x768x1>, %arg2: !migraphx.shaped<1x256x1xf32, 256x1x1>, %arg3: !migraphx.shaped<1x256x768xf32, 196608x768x1>) -> (!migraphx.shaped<1x256x768xf32, 196608x768x1>, !migraphx.shaped<1x256x768xf32, 196608x768x1>) {
        %0 = migraphx.dot %arg0, %arg1 : !migraphx.shaped<1x256x768xf32, 196608x768x1>, !migraphx.shaped<1x768x768xf32, 589824x768x1> -> !migraphx.shaped<1x256x768xf32, 196608x768x1>
        %1 = migraphx.multibroadcast %arg2 {out_lens = [1, 768, 768]} : !migraphx.shaped<1x256x1xf32, 256x1x1> -> !migraphx.shaped<1x256x768xf32, 196608x768x1>
        %2 = migraphx.add %0, %1 : !migraphx.shaped<1x256x768xf32, 196608x768x1>, !migraphx.shaped<1x256x768xf32, 196608x768x1> -> !migraphx.shaped<1x256x768xf32, 196608x768x1>
        %3 = migraphx.add %2, %arg3 : !migraphx.shaped<1x256x768xf32, 196608x768x1>, !migraphx.shaped<1x256x768xf32, 196608x768x1> -> !migraphx.shaped<1x256x768xf32, 196608x768x1>
        %4 = migraphx.relu %3 : !migraphx.shaped<1x256x768xf32, 196608x768x1> -> !migraphx.shaped<1x256x768xf32, 196608x768x1>
        return %3, %4 : !migraphx.shaped<1x256x768xf32, 196608x768x1>, !migraphx.shaped<1x256x768xf32, 196608x768x1>
    }
}
