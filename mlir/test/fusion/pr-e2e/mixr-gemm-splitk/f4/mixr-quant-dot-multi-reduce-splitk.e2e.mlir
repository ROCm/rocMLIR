// RUN: rocmlir-gen -fut quant_dot_multi_reduce --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand none -RMS_threshold=3e-3 -absDiff_threshold 7e-1 -relDiff_threshold 3e-3 -fut quant_dot_multi_reduce_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut quant_dot_multi_reduce --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_min 1 -rand_max 2 -rand_type float -RMS_threshold=3e-3 -absDiff_threshold 7e-1 -relDiff_threshold 3e-3 -fut quant_dot_multi_reduce_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen -fut quant_dot_multi_reduce --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx-linalg,highlevel -host-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -rand 1 -rand_min 1 -rand_max 2 -rand_type float -RMS_threshold=3e-3 -absDiff_threshold 7e-1 -relDiff_threshold 3e-3 -fut quant_dot_multi_reduce_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

// We need a check for each output as this test case has two outputs in it.
// CHECK: [1 1 1]
// CHECK: [1 1 1]
// CLONE: [1 1 1]
// CLONE: [1 1 1]
module {
  func.func @quant_dot_multi_reduce(%arg0: !migraphx.shaped<2x32x10x64x64xf32, 0x10x1x0x0>, %arg1: !migraphx.shaped<2x320x320xf4E2M1FN, 102400x320x1>, %arg2: !migraphx.shaped<2x320x4096xf4E2M1FN, 1310720x4096x1>, %arg3: !migraphx.shaped<2x320x10x1xf8E8M0FNU, 3200x10x1x1>, %arg4: !migraphx.shaped<2x10x1x4096xf8E8M0FNU, 40960x4096x4096x1>) -> (!migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x10x64x64xf32, 1310720x40960x4096x64x1>) attributes{rock.arch = "", rock.enable_splitk_for_tuning, rock.kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [2, 320, 10, 32]} : <2x320x10x1xf8E8M0FNU, 3200x10x1x1> -> <2x320x10x32xf8E8M0FNU, 3200x10x0x1>
    %1 = migraphx.reshape %0 {dims = [2, 320, 320]} : <2x320x10x32xf8E8M0FNU, 3200x10x0x1> -> <2x320x320xf8E8M0FNU, 3200x10x1>
    %2 = migraphx.multibroadcast %arg4 {out_dyn_dims = [], out_lens = [2, 10, 32, 4096]} : <2x10x1x4096xf8E8M0FNU, 40960x4096x4096x1> -> <2x10x32x4096xf8E8M0FNU, 40960x0x4096x1>
    %3 = migraphx.reshape %2 {dims = [2, 320, 4096]} : <2x10x32x4096xf8E8M0FNU, 40960x0x4096x1> -> <2x320x4096xf8E8M0FNU, 40960x4096x1>
    %4 = migraphx.literal(dense<2.44140629E-5> : tensor<1xf32>) : <1xf32, 0>
    %5 = migraphx.quant_dot %arg1 scaled by %1, %arg2 scaled by %3 {perf_config="v3:64,64,16,32,32,32,4,1,2,1,1"} : <2x320x320xf4E2M1FN, 102400x320x1> scaled by !migraphx.shaped<2x320x320xf8E8M0FNU, 3200x10x1>, <2x320x4096xf4E2M1FN, 1310720x4096x1> scaled by !migraphx.shaped<2x320x4096xf8E8M0FNU, 40960x4096x1> -> <2x320x4096xf32, 1310720x4096x1>
    %6 = migraphx.reshape %5 {dims = [2, 32, 10, 64, 64]} : <2x320x4096xf32, 1310720x4096x1> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %7 = migraphx.add %6, %arg0 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 0x10x1x0x0> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %8 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [2, 32, 10, 64, 64]} : <1xf32, 0> -> <2x32x10x64x64xf32, 0x0x0x0x0>
    %9 = migraphx.mul %7, %8 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 0x0x0x0x0> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %10 = migraphx.reshape %9 {dims = [2, 32, 40960, 1]} : <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x32x40960x1xf32, 1310720x40960x1x1>
    %11 = migraphx.reduce_sum %10 {axes = [2]} : <2x32x40960x1xf32, 1310720x40960x1x1> -> <2x32x1x1xf32, 32x1x1x1>
    %12 = migraphx.reshape %11 {dims = [2, 32, 1, 1, 1]} : <2x32x1x1xf32, 32x1x1x1> -> <2x32x1x1x1xf32, 32x1x1x1x1>
    return %12, %7 : !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x10x64x64xf32, 1310720x40960x4096x64x1>
  }
}
