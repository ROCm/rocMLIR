// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver --kernel-pipeline=migraphx,highlevel,gpu,binary --arch %arch --mlir-print-ir-after=rock-threadwise-gemm-lowering -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ASSEMBLY
// ASSEMBLY: amdgpu.scaled_mfma

// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -targets %arch 2>&1 | FileCheck %s --check-prefixes=GEMM
// GEMM: rock.gemm tr

// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx-linalg,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_quant_dot_fp4(%arg0: !migraphx.shaped<1x128x768xf4E2M1FN, 98304x768x1>, %arg1: !migraphx.shaped<1x768x256xf4E2M1FN, 196608x256x1>, %arg2: !migraphx.shaped<1x128x24x1xf32, 3072x24x1x1>, %arg3: !migraphx.shaped<1x24x1x256xf32, 6144x256x256x1>) -> !migraphx.shaped<1x256x128xf32, 32768x128x1> {
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 128, 24, 32]} : <1x128x24x1xf32, 3072x24x1x1> -> <1x128x24x32xf32, 3072x24x1x0>
    %1 = migraphx.reshape %0 {dims = [1, 128, 768]} : <1x128x24x32xf32, 3072x24x1x0> -> <1x128x768xf32, 98304x768x1>
    %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 24, 32, 256]} : <1x24x1x256xf32, 6144x256x256x1> -> <1x24x32x256xf32, 6144x256x0x1>
    %3 = migraphx.reshape %2 {dims = [1, 768, 256]} : <1x24x32x256xf32, 6144x256x0x1> -> <1x768x256xf32, 196608x256x1>
    %sE8A = migraphx.convert %1 : !migraphx.shaped<1x128x768xf32, 98304x768x1> to !migraphx.shaped<1x128x768xf8E8M0FNU, 98304x768x1>
    %sE8B = migraphx.convert %3 : !migraphx.shaped<1x768x256xf32, 196608x256x1> to !migraphx.shaped<1x768x256xf8E8M0FNU, 196608x256x1>
    %4 = migraphx.quant_dot %arg0 scaled by %sE8A, %arg1 scaled by %sE8B : <1x128x768xf4E2M1FN, 98304x768x1> scaled by !migraphx.shaped<1x128x768xf8E8M0FNU, 98304x768x1>, <1x768x256xf4E2M1FN, 196608x256x1> scaled by !migraphx.shaped<1x768x256xf8E8M0FNU, 196608x256x1> -> <1x128x256xf32, 32768x256x1>
    %5 = migraphx.transpose %4 {permutation = [0, 2, 1]} : <1x128x256xf32, 32768x256x1> -> <1x256x128xf32, 32768x128x1>
    return %5 : !migraphx.shaped<1x256x128xf32, 32768x128x1>
  }
}
