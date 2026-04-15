// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver --kernel-pipeline=migraphx,highlevel,gpu,binary --arch %arch --mlir-print-ir-after=rock-threadwise-gemm-lowering -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ASSEMBLY
// ASSEMBLY: amdgpu.scaled_mfma

// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx-linalg,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Same as migraphx-quant-dot-fp4.mlir but WITHOUT migraphx.convert from f32
// to f8E8M0FNU. The f32 scales are passed directly to quant_dot, testing that
// the MIGraphXToTosa lowering correctly casts them to f8E8M0FNU for
// tosa.matmul_t_block_scaled. The host-side QuantDotDecompose roundtrips f32
// scales through f8E8M0FNU to match the kernel path's behavior.
module {
  func.func @mlir_quant_dot_fp4(%arg0: !migraphx.shaped<1x2048xf4E2M1FN, 2048x1>, %arg1: !migraphx.shaped<1000x2048xf4E2M1FN, 2048x1>, %arg2: !migraphx.shaped<1x64x1xf32, 64x1x1>, %arg3: !migraphx.shaped<64x1x1000xf32, 1x1x64>, %arg4: !migraphx.shaped<1x1000xf32, 1000x1>) -> !migraphx.shaped<1x1000xf32, 1000x1> {
    %0 = migraphx.transpose %arg1 {permutation = [1, 0]} : <1000x2048xf4E2M1FN, 2048x1> -> <2048x1000xf4E2M1FN, 1x2048>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 64, 32]} : <1x64x1xf32, 64x1x1> -> <1x64x32xf32, 64x1x0>
    %2 = migraphx.reshape %1 {dims = [1, 2048]} : <1x64x32xf32, 64x1x0> -> <1x2048xf32, 2048x1>
    %3 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [64, 32, 1000]} : <64x1x1000xf32, 1x1x64> -> <64x32x1000xf32, 1x0x64>
    %4 = migraphx.reshape %3 {dims = [2048, 1000]} : <64x32x1000xf32, 1x0x64> -> <2048x1000xf32, 1000x1>
    %5 = migraphx.quant_dot %arg0 scaled by %2, %0 scaled by %4 : <1x2048xf4E2M1FN, 2048x1> scaled by !migraphx.shaped<1x2048xf32, 2048x1>, <2048x1000xf4E2M1FN, 1x2048> scaled by !migraphx.shaped<2048x1000xf32, 1000x1> -> <1x1000xf32, 1000x1>
    %6 = migraphx.add %5, %arg4 : <1x1000xf32, 1000x1>, <1x1000xf32, 1000x1> -> <1x1000xf32, 1000x1>
    return %6 : !migraphx.shaped<1x1000xf32, 1000x1>
  }
}
