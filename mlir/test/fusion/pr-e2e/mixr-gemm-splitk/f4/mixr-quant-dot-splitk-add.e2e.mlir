// RUN: rocmlir-gen -fut quant_dot_splitk_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand none -fut quant_dot_splitk_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// RUN: rocmlir-gen -fut quant_dot_splitk_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut quant_dot_splitk_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
module {
  // CHECK: [1 1 1]
  // CLONE: [1 1 1]
  func.func @quant_dot_splitk_add(%arg0: !migraphx.shaped<1x128x256xf4E2M1FN, 32768x256x1>, %arg1: !migraphx.shaped<1x256x128xf4E2M1FN, 32768x128x1>, %arg2: !migraphx.shaped<1x128x8x1xf8E8M0FNU, 1024x8x1x1>, %arg3: !migraphx.shaped<1x8x1x128xf8E8M0FNU, 1024x128x128x1>, %arg4: !migraphx.shaped<1x128x128xf32, 16384x128x1>) -> !migraphx.shaped<1x128x128xf32, 16384x128x1> attributes{rock.arch = "", rock.enable_splitk_for_tuning, rock.kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 128, 8, 32]} : <1x128x8x1xf8E8M0FNU, 1024x8x1x1> -> <1x128x8x32xf8E8M0FNU, 1024x8x0x1>
    %1 = migraphx.reshape %0 {dims = [1, 128, 256]} : <1x128x8x32xf8E8M0FNU, 1024x8x0x1> -> <1x128x256xf8E8M0FNU, 1024x8x1>
    %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 8, 32, 128]} : <1x8x1x128xf8E8M0FNU, 1024x128x128x1> -> <1x8x32x128xf8E8M0FNU, 1024x0x128x1>
    %3 = migraphx.reshape %2 {dims = [1, 256, 128]} : <1x8x32x128xf8E8M0FNU, 1024x0x128x1> -> <1x256x128xf8E8M0FNU, 1024x128x1>
    %4 = migraphx.quant_dot %arg0 scaled by %1, %arg1 scaled by %3 {perf_config="v3:64,64,16,32,32,32,4,1,2,1,1"} : <1x128x256xf4E2M1FN, 32768x256x1> scaled by !migraphx.shaped<1x128x256xf8E8M0FNU, 1024x8x1>, <1x256x128xf4E2M1FN, 32768x128x1> scaled by !migraphx.shaped<1x256x128xf8E8M0FNU, 1024x128x1> -> <1x128x128xf32, 16384x128x1>
    %5 = migraphx.add %4, %arg4 {} : <1x128x128xf32, 16384x128x1>, <1x128x128xf32, 16384x128x1> -> <1x128x128xf32, 16384x128x1>
    return %5 : !migraphx.shaped<1x128x128xf32, 16384x128x1>
  }
}

