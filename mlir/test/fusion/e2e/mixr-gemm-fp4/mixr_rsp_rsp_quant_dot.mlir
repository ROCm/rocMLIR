// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// EMITKEY: -t f4E2M1FN -out_datatype f32 -transA false -transB false -scaledGemm -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -transScaleA false -transScaleB false -g 1 -m 256 -n 50272 -k 768 

// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver --kernel-pipeline=migraphx,highlevel,gpu,binary --arch %arch --mlir-print-ir-after=rock-threadwise-gemm-lowering -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ASSEMBLY
// ASSEMBLY: amdgpu.scaled_mfma

// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx-linalg,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_quant_dot_fp4 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_fp4_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_quant_dot_fp4(%arg0: !migraphx.shaped<1x256x768xf4E2M1FN, 196608x768x1>, %arg1: !migraphx.shaped<768x50272xf4E2M1FN, 50272x1>, %arg2: !migraphx.shaped<1x8x1x768xf32, 6144x768x768x1>, %arg3: !migraphx.shaped<1x768x1571x1xf32, 1206528x1571x1x1>) -> !migraphx.shaped<1x256x50272xf32, 12869632x50272x1> {
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [1, 768, 50272]} : <768x50272xf4E2M1FN, 50272x1> -> <1x768x50272xf4E2M1FN, 0x50272x1>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 8, 32, 768]} : <1x8x1x768xf32, 6144x768x768x1> -> <1x8x32x768xf32, 6144x768x0x1>
    %2 = migraphx.reshape %1 {dims = [1, 256, 768]} : <1x8x32x768xf32, 6144x768x0x1> -> <1x256x768xf32, 196608x768x1>
    %3 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 768, 1571, 32]} : <1x768x1571x1xf32, 1206528x1571x1x1> -> <1x768x1571x32xf32, 1206528x1571x1x0>
    %4 = migraphx.reshape %3 {dims = [1, 768, 50272]} : <1x768x1571x32xf32, 1206528x1571x1x0> -> <1x768x50272xf32, 38608896x50272x1>
    %sE8A = migraphx.convert %2 : !migraphx.shaped<1x256x768xf32, 196608x768x1> to !migraphx.shaped<1x256x768xf8E8M0FNU, 196608x768x1>
    %sE8B = migraphx.convert %4 : !migraphx.shaped<1x768x50272xf32, 38608896x50272x1> to !migraphx.shaped<1x768x50272xf8E8M0FNU, 38608896x50272x1>
    %5 = migraphx.quant_dot %arg0 scaled by %sE8A, %0 scaled by %sE8B : <1x256x768xf4E2M1FN, 196608x768x1> scaled by !migraphx.shaped<1x256x768xf8E8M0FNU, 196608x768x1>, <1x768x50272xf4E2M1FN, 0x50272x1> scaled by !migraphx.shaped<1x768x50272xf8E8M0FNU, 38608896x50272x1> -> <1x256x50272xf32, 12869632x50272x1>
    return %5 : !migraphx.shaped<1x256x50272xf32, 12869632x50272x1>
  }
}
