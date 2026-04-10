// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver --kernel-pipeline=migraphx,highlevel,gpu,binary --arch %arch --mlir-print-ir-after=rock-threadwise-gemm-lowering -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ASSEMBLY
// ASSEMBLY: amdgpu.scaled_mfma

// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx-linalg,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_quant_dot_fp4(%arg0: !migraphx.shaped<256x768xf4E2M1FN, 768x1>, %arg1: !migraphx.shaped<3072x768xf4E2M1FN, 768x1>, %arg2: !migraphx.shaped<256x24x1xf32, 24x1x1>, %arg3: !migraphx.shaped<24x1x3072xf32, 1x1x24>, %arg4: !migraphx.shaped<96x32xf32, 32x1>) -> !migraphx.shaped<256x96x32xf32, 3072x32x1> {
    %0 = migraphx.transpose %arg1 {permutation = [1, 0]} : <3072x768xf4E2M1FN, 768x1> -> <768x3072xf4E2M1FN, 1x768>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [256, 24, 32]} : <256x24x1xf32, 24x1x1> -> <256x24x32xf32, 24x1x0>
    %2 = migraphx.reshape %1 {dims = [256, 768]} : <256x24x32xf32, 24x1x0> -> <256x768xf32, 768x1>
    %3 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [24, 32, 3072]} : <24x1x3072xf32, 1x1x24> -> <24x32x3072xf32, 1x0x24>
    %4 = migraphx.reshape %3 {dims = [768, 3072]} : <24x32x3072xf32, 1x0x24> -> <768x3072xf32, 3072x1>
    %sE8A = migraphx.convert %2 : !migraphx.shaped<256x768xf32, 768x1> to !migraphx.shaped<256x768xf8E8M0FNU, 768x1>
    %sE8B = migraphx.convert %4 : !migraphx.shaped<768x3072xf32, 3072x1> to !migraphx.shaped<768x3072xf8E8M0FNU, 3072x1>
    %5 = migraphx.quant_dot %arg0 scaled by %sE8A, %0 scaled by %sE8B : <256x768xf4E2M1FN, 768x1> scaled by !migraphx.shaped<256x768xf8E8M0FNU, 768x1>, <768x3072xf4E2M1FN, 1x768> scaled by !migraphx.shaped<768x3072xf8E8M0FNU, 3072x1> -> <256x3072xf32, 3072x1>
    %6 = migraphx.reshape %5 {dims = [256, 96, 32]} : <256x3072xf32, 3072x1> -> <256x96x32xf32, 3072x32x1>
    %7 = migraphx.broadcast %arg4 {axis = 1 : i64, out_lens = [256, 96, 32]} : <96x32xf32, 32x1> -> <256x96x32xf32, 0x32x1>
    %8 = migraphx.add %6, %7 : <256x96x32xf32, 3072x32x1>, <256x96x32xf32, 0x32x1> -> <256x96x32xf32, 3072x32x1>
    %9 = migraphx.relu %8 : <256x96x32xf32, 3072x32x1> -> <256x96x32xf32, 3072x32x1>
    return %9 : !migraphx.shaped<256x96x32xf32, 3072x32x1>
  }
}
