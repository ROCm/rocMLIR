// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver --kernel-pipeline=migraphx,highlevel,gpu,binary --arch %arch --mlir-print-ir-after=rock-threadwise-gemm-lowering -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ASSEMBLY
// ASSEMBLY: amdgpu.scaled_mfma

// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx-linalg,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_quant_dot_fp4(%arg0: !migraphx.shaped<1x256x768xf4E2M1FN, 196608x768x1>, %arg1: !migraphx.shaped<768x768xf4E2M1FN, 768x1>, %arg2: !migraphx.shaped<1x8x1x768xf32, 6144x768x768x1>, %arg3: !migraphx.shaped<1x768x24x1xf32, 18432x24x1x1>, %arg4: !migraphx.shaped<768xf32, 1>) -> !migraphx.shaped<1x8x32x768xf32, 196608x24576x768x1> {
    %0 = migraphx.literal(dense<1.250000e-01> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [1, 768, 768]} : <768x768xf4E2M1FN, 768x1> -> <1x768x768xf4E2M1FN, 0x768x1>
    %2 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 8, 32, 768]} : <1x8x1x768xf32, 6144x768x768x1> -> <1x8x32x768xf32, 6144x768x0x1>
    %3 = migraphx.reshape %2 {dims = [1, 256, 768]} : <1x8x32x768xf32, 6144x768x0x1> -> <1x256x768xf32, 196608x768x1>
    %4 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 768, 24, 32]} : <1x768x24x1xf32, 18432x24x1x1> -> <1x768x24x32xf32, 18432x24x1x0>
    %5 = migraphx.reshape %4 {dims = [1, 768, 768]} : <1x768x24x32xf32, 18432x24x1x0> -> <1x768x768xf32, 589824x768x1>
    %sE8A = migraphx.convert %3 : !migraphx.shaped<1x256x768xf32, 196608x768x1> to !migraphx.shaped<1x256x768xf8E8M0FNU, 196608x768x1>
    %sE8B = migraphx.convert %5 : !migraphx.shaped<1x768x768xf32, 589824x768x1> to !migraphx.shaped<1x768x768xf8E8M0FNU, 589824x768x1>
    %6 = migraphx.quant_dot %arg0 scaled by %sE8A, %1 scaled by %sE8B : <1x256x768xf4E2M1FN, 196608x768x1> scaled by !migraphx.shaped<1x256x768xf8E8M0FNU, 196608x768x1>, <1x768x768xf4E2M1FN, 0x768x1> scaled by !migraphx.shaped<1x768x768xf8E8M0FNU, 589824x768x1> -> <1x256x768xf32, 196608x768x1>
    %7 = migraphx.reshape %6 {dims = [1, 8, 32, 768]} : <1x256x768xf32, 196608x768x1> -> <1x8x32x768xf32, 196608x24576x768x1>
    %8 = migraphx.broadcast %arg4 {axis = 3 : i64, out_lens = [1, 8, 32, 768]} : <768xf32, 1> -> <1x8x32x768xf32, 0x0x0x1>
    %9 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 8, 32, 768]} : <1xf32, 0> -> <1x8x32x768xf32, 0x0x0x0>
    %10 = migraphx.mul %9, %7 : <1x8x32x768xf32, 0x0x0x0>, <1x8x32x768xf32, 196608x24576x768x1> -> <1x8x32x768xf32, 196608x24576x768x1>
    %11 = migraphx.add %10, %8 : <1x8x32x768xf32, 196608x24576x768x1>, <1x8x32x768xf32, 0x0x0x1> -> <1x8x32x768xf32, 196608x24576x768x1>
    return %11 : !migraphx.shaped<1x8x32x768xf32, 196608x24576x768x1>
  }
}
