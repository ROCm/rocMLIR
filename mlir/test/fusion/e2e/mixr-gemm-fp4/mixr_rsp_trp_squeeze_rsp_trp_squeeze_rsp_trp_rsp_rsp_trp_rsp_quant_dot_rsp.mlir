// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver --kernel-pipeline=migraphx,highlevel,gpu,binary --arch %arch --mlir-print-ir-after=rock-threadwise-gemm-lowering -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ASSEMBLY
// ASSEMBLY: amdgpu.scaled_mfma

// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx-linalg,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_quant_dot_fp4(%arg0: !migraphx.shaped<1x256x128xf4E2M1FN, 32768x128x1>, %arg1: !migraphx.shaped<1x256x128xf4E2M1FN, 32768x128x1>, %arg2: !migraphx.shaped<1x8x1x128xf32, 1024x128x128x1>, %arg3: !migraphx.shaped<1x8x1x128xf32, 1024x128x128x1>) -> !migraphx.shaped<1x2x8x32x256xf32, 131072x65536x8192x256x1> {
    %0 = migraphx.reshape %arg0 {dims = [1, 256, 2, 64]} : <1x256x128xf4E2M1FN, 32768x128x1> -> <1x256x2x64xf4E2M1FN, 32768x128x64x1>
    %1 = migraphx.transpose %0 {permutation = [2, 0, 1, 3]} : <1x256x2x64xf4E2M1FN, 32768x128x64x1> -> <2x1x256x64xf4E2M1FN, 64x32768x128x1>
    %2 = migraphx.reshape %1 {dims = [2, 256, 64]} : <2x1x256x64xf4E2M1FN, 64x32768x128x1> -> <2x256x64xf4E2M1FN, 16384x64x1>
    %3 = migraphx.reshape %arg1 {dims = [1, 256, 2, 64]} : <1x256x128xf4E2M1FN, 32768x128x1> -> <1x256x2x64xf4E2M1FN, 32768x128x64x1>
    %4 = migraphx.transpose %3 {permutation = [2, 3, 0, 1]} : <1x256x2x64xf4E2M1FN, 32768x128x64x1> -> <2x64x1x256xf4E2M1FN, 64x1x32768x128>
    %5 = migraphx.reshape %4 {dims = [2, 64, 256]} : <2x64x1x256xf4E2M1FN, 64x1x32768x128> -> <2x64x256xf4E2M1FN, 16384x256x1>
    %6 = migraphx.reshape %arg2 {dims = [1, 8, 1, 2, 32, 2]} : <1x8x1x128xf32, 1024x128x128x1> -> <1x8x1x2x32x2xf32, 1024x128x128x64x32x1>
    %7 = migraphx.transpose %6 {permutation = [3, 0, 1, 4, 5, 2]} : <1x8x1x2x32x2xf32, 1024x128x128x64x32x1> -> <2x1x8x32x2x1xf32, 64x1024x128x32x1x128>
    %8 = migraphx.multibroadcast %7 {out_dyn_dims = [], out_lens = [2, 1, 8, 32, 2, 32]} : <2x1x8x32x2x1xf32, 64x1024x128x32x1x128> -> <2x1x8x32x2x32xf32, 64x1024x128x32x1x0>
    %9 = migraphx.reshape %8 {dims = [2, 256, 64]} : <2x1x8x32x2x32xf32, 64x1024x128x32x1x0> -> <2x256x64xf32, 16384x64x1>
    %10 = migraphx.reshape %arg3 {dims = [1, 8, 1, 2, 32, 2]} : <1x8x1x128xf32, 1024x128x128x1> -> <1x8x1x2x32x2xf32, 1024x128x128x64x32x1>
    %11 = migraphx.transpose %10 {permutation = [3, 5, 0, 1, 4, 2]} : <1x8x1x2x32x2xf32, 1024x128x128x64x32x1> -> <2x2x1x8x32x1xf32, 64x1x1024x128x32x128>
    %12 = migraphx.multibroadcast %11 {out_dyn_dims = [], out_lens = [2, 2, 32, 8, 32, 1]} : <2x2x1x8x32x1xf32, 64x1x1024x128x32x128> -> <2x2x32x8x32x1xf32, 64x1x0x128x32x128>
    %13 = migraphx.reshape %12 {dims = [2, 64, 256]} : <2x2x32x8x32x1xf32, 64x1x0x128x32x128> -> <2x64x256xf32, 16384x256x1>
    %sE8A = migraphx.convert %9 : !migraphx.shaped<2x256x64xf32, 16384x64x1> to !migraphx.shaped<2x256x64xf8E8M0FNU, 16384x64x1>
    %sE8B = migraphx.convert %13 : !migraphx.shaped<2x64x256xf32, 16384x256x1> to !migraphx.shaped<2x64x256xf8E8M0FNU, 16384x256x1>
    %14 = migraphx.quant_dot %2 scaled by %sE8A, %5 scaled by %sE8B : <2x256x64xf4E2M1FN, 16384x64x1> scaled by !migraphx.shaped<2x256x64xf8E8M0FNU, 16384x64x1>, <2x64x256xf4E2M1FN, 16384x256x1> scaled by !migraphx.shaped<2x64x256xf8E8M0FNU, 16384x256x1> -> <2x256x256xf32, 65536x256x1>
    %15 = migraphx.reshape %14 {dims = [1, 2, 8, 32, 256]} : <2x256x256xf32, 65536x256x1> -> <1x2x8x32x256xf32, 131072x65536x8192x256x1>
    return %15 : !migraphx.shaped<1x2x8x32x256xf32, 131072x65536x8192x256x1>
  }
}
