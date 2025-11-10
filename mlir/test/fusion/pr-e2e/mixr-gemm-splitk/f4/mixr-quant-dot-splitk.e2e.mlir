// RUN: rocmlir-gen -fut quant_dot_splitk --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand none -fut quant_dot_splitk_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
module {
  // CHECK: [1 1 1]
  func.func @quant_dot_splitk(%arg0: !migraphx.shaped<1x64x128xf4E2M1FN, 8192x128x1>, %arg1: !migraphx.shaped<1x128x64xf4E2M1FN, 8192x64x1>, %arg2: !migraphx.shaped<1x2x1x128xf8E8M0FNU, 256x128x128x1>, %arg3: !migraphx.shaped<1x128x2x1xf8E8M0FNU, 256x2x1x1>) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> attributes{arch = "gfx950", enable_splitk_for_tuning, kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 2, 32, 128]} : <1x2x1x128xf8E8M0FNU, 256x128x128x1> -> <1x2x32x128xf8E8M0FNU, 256x128x0x1>
    %1 = migraphx.reshape %0 {dims = [1, 64, 128]} : <1x2x32x128xf8E8M0FNU, 256x128x0x1> -> <1x64x128xf8E8M0FNU, 8192x128x1>
    %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 128, 2, 32]} : <1x128x2x1xf8E8M0FNU, 256x2x1x1> -> <1x128x2x32xf8E8M0FNU, 256x2x1x0>
    %3 = migraphx.reshape %2 {dims = [1, 128, 64]} : <1x128x2x32xf8E8M0FNU, 256x2x1x0> -> <1x128x64xf8E8M0FNU, 8192x64x1>
    %4 = migraphx.quant_dot %arg0 scaled by %1, %arg1 scaled by %3 {perf_config="v3:64,64,16,32,32,32,2,1,2,1,1"} : <1x64x128xf4E2M1FN, 8192x128x1> scaled by !migraphx.shaped<1x64x128xf8E8M0FNU, 8192x128x1>, <1x128x64xf4E2M1FN, 8192x64x1> scaled by !migraphx.shaped<1x128x64xf8E8M0FNU, 8192x64x1> -> <1x64x64xf32, 4096x64x1>
    return %4 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
  }
}

