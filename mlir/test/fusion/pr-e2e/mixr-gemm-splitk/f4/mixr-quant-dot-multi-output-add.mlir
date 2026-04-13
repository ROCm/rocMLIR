// RUN: rocmlir-gen -fut quant_dot_multi_output_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand none -fut quant_dot_multi_output_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut quant_dot_multi_output_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut quant_dot_multi_output_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// We need a check for each output as this test case has two outputs in it.
// CHECK: [1 1 1]
// CHECK: [1 1 1]
// CLONE: [1 1 1]
// CLONE: [1 1 1]
module {
  func.func @quant_dot_multi_output_add(%arg0: !migraphx.shaped<1x64x128xf4E2M1FN, 8192x128x1>, %arg1: !migraphx.shaped<1x128x64xf4E2M1FN, 8192x64x1>, %arg2: !migraphx.shaped<1x64x4x1xf8E8M0FNU, 256x4x1x1>, %arg3: !migraphx.shaped<1x4x1x64xf8E8M0FNU, 256x64x64x1>) -> (!migraphx.shaped<1x64x64xf32, 4096x64x1>, !migraphx.shaped<1x64x64xf32, 4096x64x1>) attributes{rock.arch = "", rock.enable_splitk_for_tuning, rock.kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 64, 4, 32]} : <1x64x4x1xf8E8M0FNU, 256x4x1x1> -> <1x64x4x32xf8E8M0FNU, 256x4x0x1>
    %1 = migraphx.reshape %0 {dims = [1, 64, 128]} : <1x64x4x32xf8E8M0FNU, 256x4x0x1> -> <1x64x128xf8E8M0FNU, 256x4x1>
    %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 4, 32, 64]} : <1x4x1x64xf8E8M0FNU, 256x64x64x1> -> <1x4x32x64xf8E8M0FNU, 256x0x64x1>
    %3 = migraphx.reshape %2 {dims = [1, 128, 64]} : <1x4x32x64xf8E8M0FNU, 256x0x64x1> -> <1x128x64xf8E8M0FNU, 256x64x1>
    %4 = migraphx.literal(dense<1.0> : tensor<1xf32>) : <1xf32, 0>
    %5 = migraphx.literal(dense<2.0> : tensor<1xf32>) : <1xf32, 0>
    %6 = migraphx.quant_dot %arg0 scaled by %1, %arg1 scaled by %3 {perf_config="v3:64,64,16,32,32,32,3,1,2,1,1"} : <1x64x128xf4E2M1FN, 8192x128x1> scaled by !migraphx.shaped<1x64x128xf8E8M0FNU, 256x4x1>, <1x128x64xf4E2M1FN, 8192x64x1> scaled by !migraphx.shaped<1x128x64xf8E8M0FNU, 256x64x1> -> <1x64x64xf32, 4096x64x1>
    %7 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [1, 64, 64]} : <1xf32, 0> -> <1x64x64xf32, 0x0x0>
    %8 = migraphx.multibroadcast %5 {out_dyn_dims = [], out_lens = [1, 64, 64]} : <1xf32, 0> -> <1x64x64xf32, 0x0x0>
    %9 = migraphx.add %6, %7 : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf32, 4096x64x1>
    %10 = migraphx.add %6, %8 : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf32, 4096x64x1>
    return %9, %10 : !migraphx.shaped<1x64x64xf32, 4096x64x1>, !migraphx.shaped<1x64x64xf32, 4096x64x1>
  }
}

