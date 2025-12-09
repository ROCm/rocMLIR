// RUN: rocmlir-driver --kernel-pipeline=migraphx %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test lowering of migraphx.quant_dot to tosa operations.
// Note: Scaled quant_dot with block_size=32 lowering to tosa.matmul_t_block_scaled
// is tested in the E2E tests (mixr-quant-dot-*.mlir) due to type conversion
// complexities with broadcast strides.
//===----------------------------------------------------------------------===//

// ============================================================================
// TEST 1: quant_dot without scales - should use regular tosa.matmul
// ============================================================================
// CHECK-LABEL: quant_dot_no_scales_fallback
// No matmul_t_block_scaled, should use regular matmul
// CHECK-NOT: tosa.matmul_t_block_scaled
// CHECK: tosa.matmul
// CHECK-SAME: acc_type = i32
func.func @quant_dot_no_scales_fallback(
    %arg0: !migraphx.shaped<1x64x128xi8, 8192x128x1>,
    %arg1: !migraphx.shaped<1x128x64xi8, 8192x64x1>
) -> !migraphx.shaped<1x64x64xi32, 4096x64x1> {
  %0 = migraphx.quant_dot %arg0, %arg1
    : !migraphx.shaped<1x64x128xi8, 8192x128x1>,
      !migraphx.shaped<1x128x64xi8, 8192x64x1>
    -> !migraphx.shaped<1x64x64xi32, 4096x64x1>
  return %0 : !migraphx.shaped<1x64x64xi32, 4096x64x1>
}

// ============================================================================
// TEST 2: FP8 quant_dot without scales - uses regular tosa.matmul
// ============================================================================
// CHECK-LABEL: quant_dot_fp8_no_scales
// CHECK-NOT: tosa.matmul_t_block_scaled
// CHECK: tosa.matmul
// CHECK-SAME: acc_type = f32
func.func @quant_dot_fp8_no_scales(
    %arg0: !migraphx.shaped<1x64x128xf8E4M3FNUZ, 8192x128x1>,
    %arg1: !migraphx.shaped<1x128x64xf8E4M3FNUZ, 8192x64x1>
) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> {
  %0 = migraphx.quant_dot %arg0, %arg1
    : !migraphx.shaped<1x64x128xf8E4M3FNUZ, 8192x128x1>,
      !migraphx.shaped<1x128x64xf8E4M3FNUZ, 8192x64x1>
    -> !migraphx.shaped<1x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
}

// ============================================================================
// TEST 3: Batched quant_dot without scales
// ============================================================================
// CHECK-LABEL: quant_dot_batched_no_scales
// CHECK-NOT: tosa.matmul_t_block_scaled
// CHECK: tosa.matmul
// CHECK-SAME: acc_type = f32
func.func @quant_dot_batched_no_scales(
    %arg0: !migraphx.shaped<8x64x128xf8E4M3FNUZ, 8192x128x1>,
    %arg1: !migraphx.shaped<8x128x64xf8E4M3FNUZ, 8192x64x1>
) -> !migraphx.shaped<8x64x64xf32, 4096x64x1> {
  %0 = migraphx.quant_dot %arg0, %arg1
    : !migraphx.shaped<8x64x128xf8E4M3FNUZ, 8192x128x1>,
      !migraphx.shaped<8x128x64xf8E4M3FNUZ, 8192x64x1>
    -> !migraphx.shaped<8x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<8x64x64xf32, 4096x64x1>
}

// ============================================================================
// TEST 4: Scaled quant_dot with blockSize=32 - should use tosa.matmul_t_block_scaled
// Scales are broadcasted to match A/B shapes, then need to be un-broadcasted
// ============================================================================
// CHECK-LABEL: quant_dot_with_scales
// CHECK: tosa.matmul_t_block_scaled
// CHECK-SAME: block_size = BLOCK_SIZE_32
func.func @quant_dot_with_scales(
    %arg0: !migraphx.shaped<1x64x128xf4E2M1FN, 8192x128x1>,
    %arg1: !migraphx.shaped<1x128x64xf4E2M1FN, 8192x64x1>,
    %arg2: !migraphx.shaped<1x64x4x1xf8E8M0FNU, 256x4x1x1>,
    %arg3: !migraphx.shaped<1x4x1x64xf8E8M0FNU, 256x64x64x1>
) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> attributes {kernel} {
  // Broadcast and reshape scaleA: [1,64,4,1] -> [1,64,4,32] -> [1,64,128]
  %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 64, 4, 32]}
    : <1x64x4x1xf8E8M0FNU, 256x4x1x1> -> <1x64x4x32xf8E8M0FNU, 256x4x1x0>
  %1 = migraphx.reshape %0 {dims = [1, 64, 128]}
    : <1x64x4x32xf8E8M0FNU, 256x4x1x0> -> <1x64x128xf8E8M0FNU, 8192x128x1>
  // Broadcast and reshape scaleB: [1,4,1,64] -> [1,4,32,64] -> [1,128,64]
  %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 4, 32, 64]}
    : <1x4x1x64xf8E8M0FNU, 256x64x64x1> -> <1x4x32x64xf8E8M0FNU, 256x64x0x1>
  %3 = migraphx.reshape %2 {dims = [1, 128, 64]}
    : <1x4x32x64xf8E8M0FNU, 256x64x0x1> -> <1x128x64xf8E8M0FNU, 8192x64x1>
  // Scaled quant_dot
  %4 = migraphx.quant_dot %arg0 scaled by %1, %arg1 scaled by %3
    : <1x64x128xf4E2M1FN, 8192x128x1> scaled by !migraphx.shaped<1x64x128xf8E8M0FNU, 8192x128x1>,
      <1x128x64xf4E2M1FN, 8192x64x1> scaled by !migraphx.shaped<1x128x64xf8E8M0FNU, 8192x64x1>
    -> <1x64x64xf32, 4096x64x1>
  return %4 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
}

// ============================================================================
// TEST 5: Scaled quant_dot with perf_config attribute propagation
// ============================================================================
// CHECK-LABEL: quant_dot_with_scales_perf_config
// CHECK: tosa.matmul_t_block_scaled
// CHECK-SAME: block_size = BLOCK_SIZE_32
// CHECK-SAME: perf_config = "test_perf_config"
func.func @quant_dot_with_scales_perf_config(
    %arg0: !migraphx.shaped<1x64x128xf4E2M1FN, 8192x128x1>,
    %arg1: !migraphx.shaped<1x128x64xf4E2M1FN, 8192x64x1>,
    %arg2: !migraphx.shaped<1x64x4x1xf8E8M0FNU, 256x4x1x1>,
    %arg3: !migraphx.shaped<1x4x1x64xf8E8M0FNU, 256x64x64x1>
) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> attributes {kernel} {
  // Broadcast and reshape scaleA: [1,64,4,1] -> [1,64,4,32] -> [1,64,128]
  %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 64, 4, 32]}
    : <1x64x4x1xf8E8M0FNU, 256x4x1x1> -> <1x64x4x32xf8E8M0FNU, 256x4x1x0>
  %1 = migraphx.reshape %0 {dims = [1, 64, 128]}
    : <1x64x4x32xf8E8M0FNU, 256x4x1x0> -> <1x64x128xf8E8M0FNU, 8192x128x1>
  // Broadcast and reshape scaleB: [1,4,1,64] -> [1,4,32,64] -> [1,128,64]
  %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 4, 32, 64]}
    : <1x4x1x64xf8E8M0FNU, 256x64x64x1> -> <1x4x32x64xf8E8M0FNU, 256x64x0x1>
  %3 = migraphx.reshape %2 {dims = [1, 128, 64]}
    : <1x4x32x64xf8E8M0FNU, 256x64x0x1> -> <1x128x64xf8E8M0FNU, 8192x64x1>
  // Scaled quant_dot with perf_config
  %4 = migraphx.quant_dot %arg0 scaled by %1, %arg1 scaled by %3 {perf_config = "test_perf_config"}
    : <1x64x128xf4E2M1FN, 8192x128x1> scaled by !migraphx.shaped<1x64x128xf8E8M0FNU, 8192x128x1>,
      <1x128x64xf4E2M1FN, 8192x64x1> scaled by !migraphx.shaped<1x128x64xf8E8M0FNU, 8192x64x1>
    -> <1x64x64xf32, 4096x64x1>
  return %4 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
}
