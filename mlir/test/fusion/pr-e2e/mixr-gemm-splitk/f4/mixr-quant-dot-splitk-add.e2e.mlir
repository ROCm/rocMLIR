// RUN: rocmlir-gen -fut quant_dot_splitk_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand none -fut quant_dot_splitk_add_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
module {
  // CHECK: [1 1 1]
  func.func @quant_dot_splitk_add(%arg0: !migraphx.shaped<1x128x256xf4E2M1FN, 32768x256x1>, %arg1: !migraphx.shaped<1x256x128xf4E2M1FN, 32768x128x1>, %arg2: !migraphx.shaped<1x4x1x256xf8E8M0FNU, 1024x256x256x1>, %arg3: !migraphx.shaped<1x256x4x1xf8E8M0FNU, 1024x4x1x1>, %arg4: !migraphx.shaped<1x128x128xf32, 16384x128x1>) -> !migraphx.shaped<1x128x128xf32, 16384x128x1> attributes{arch = "", enable_splitk_for_tuning, kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 4, 32, 256]} : <1x4x1x256xf8E8M0FNU, 1024x256x256x1> -> <1x4x32x256xf8E8M0FNU, 1024x256x0x1>
    %1 = migraphx.reshape %0 {dims = [1, 128, 256]} : <1x4x32x256xf8E8M0FNU, 1024x256x0x1> -> <1x128x256xf8E8M0FNU, 32768x256x1>
    %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 256, 4, 32]} : <1x256x4x1xf8E8M0FNU, 1024x4x1x1> -> <1x256x4x32xf8E8M0FNU, 1024x4x1x0>
    %3 = migraphx.reshape %2 {dims = [1, 256, 128]} : <1x256x4x32xf8E8M0FNU, 1024x4x1x0> -> <1x256x128xf8E8M0FNU, 32768x128x1>
    %4 = migraphx.quant_dot %arg0 scaled by %1, %arg1 scaled by %3 {perf_config="v3:64,64,16,32,32,32,4,1,2,1,1"} : <1x128x256xf4E2M1FN, 32768x256x1> scaled by !migraphx.shaped<1x128x256xf8E8M0FNU, 32768x256x1>, <1x256x128xf4E2M1FN, 32768x128x1> scaled by !migraphx.shaped<1x256x128xf8E8M0FNU, 32768x128x1> -> <1x128x128xf32, 16384x128x1>
    %5 = migraphx.add %4, %arg4 {} : <1x128x128xf32, 16384x128x1>, <1x128x128xf32, 16384x128x1> -> <1x128x128xf32, 16384x128x1>
    return %5 : !migraphx.shaped<1x128x128xf32, 16384x128x1>
  }
}

