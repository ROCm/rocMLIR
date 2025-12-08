// RUN: rocmlir-driver --kernel-pipeline=migraphx %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test lowering of migraphx.quant_dot with scales through the decomposition path.
// Scaled quant_dot operations are decomposed into convert+mul+dot by the
// migraphx-transform pass, which then get lowered to tosa.cast+tosa.mul+tosa.matmul.
//===----------------------------------------------------------------------===//

// ============================================================================
// TEST 1: Basic 3D tensor test with FP4 data
// ============================================================================
// CHECK-LABEL: quant_dot_scaled_3d_fp4
// Decomposed path: cast data, cast scale (if needed), mul, then matmul
// CHECK: tosa.cast
// CHECK: tosa.mul
// CHECK: tosa.matmul
// CHECK-NOT: tosa.matmul_t_block_scaled
func.func @quant_dot_scaled_3d_fp4(
    %arg0: !migraphx.shaped<1x64x128xf4E2M1FN, 8192x128x1>,
    %arg1: !migraphx.shaped<1x128x64xf4E2M1FN, 8192x64x1>,
    %arg2: !migraphx.shaped<1x64x4x1xf8E8M0FNU, 256x4x1x1>,
    %arg3: !migraphx.shaped<1x4x1x64xf8E8M0FNU, 256x64x64x1>
) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> {
  %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 64, 4, 32]} : <1x64x4x1xf8E8M0FNU, 256x4x1x1> -> <1x64x4x32xf8E8M0FNU, 256x4x0x1>
  %1 = migraphx.reshape %0 {dims = [1, 64, 128]} : <1x64x4x32xf8E8M0FNU, 256x4x0x1> -> <1x64x128xf8E8M0FNU, 256x4x1>
  %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [1, 4, 32, 64]} : <1x4x1x64xf8E8M0FNU, 256x64x64x1> -> <1x4x32x64xf8E8M0FNU, 256x0x64x1>
  %3 = migraphx.reshape %2 {dims = [1, 128, 64]} : <1x4x32x64xf8E8M0FNU, 256x0x64x1> -> <1x128x64xf8E8M0FNU, 256x64x1>
  
  %4 = migraphx.quant_dot %arg0 scaled by %1, %arg1 scaled by %3
    : !migraphx.shaped<1x64x128xf4E2M1FN, 8192x128x1>
        scaled by !migraphx.shaped<1x64x128xf8E8M0FNU, 256x4x1>,
      !migraphx.shaped<1x128x64xf4E2M1FN, 8192x64x1>
        scaled by !migraphx.shaped<1x128x64xf8E8M0FNU, 256x64x1>
    -> !migraphx.shaped<1x64x64xf32, 4096x64x1>
  return %4 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
}

// ============================================================================
// TEST 2: 4D tensor test - requires batch axis flattening
// ============================================================================
// CHECK-LABEL: quant_dot_scaled_4d_batch_flatten
// Decomposed path with batch flattening
// CHECK: tosa.cast
// CHECK: tosa.mul
// CHECK: tosa.matmul
// CHECK-NOT: tosa.matmul_t_block_scaled
func.func @quant_dot_scaled_4d_batch_flatten(
    %arg0: !migraphx.shaped<2x3x64x128xf4E2M1FN, 24576x8192x128x1>,
    %arg1: !migraphx.shaped<2x3x128x64xf4E2M1FN, 24576x8192x64x1>,
    %arg2: !migraphx.shaped<2x3x64x4x1xf8E8M0FNU, 768x256x4x1x1>,
    %arg3: !migraphx.shaped<2x3x4x1x64xf8E8M0FNU, 768x256x64x64x1>
) -> !migraphx.shaped<2x3x64x64xf32, 12288x4096x64x1> {
  %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [2, 3, 64, 4, 32]} : <2x3x64x4x1xf8E8M0FNU, 768x256x4x1x1> -> <2x3x64x4x32xf8E8M0FNU, 768x256x4x0x1>
  %1 = migraphx.reshape %0 {dims = [2, 3, 64, 128]} : <2x3x64x4x32xf8E8M0FNU, 768x256x4x0x1> -> <2x3x64x128xf8E8M0FNU, 768x256x4x1>
  %2 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [2, 3, 4, 32, 64]} : <2x3x4x1x64xf8E8M0FNU, 768x256x64x64x1> -> <2x3x4x32x64xf8E8M0FNU, 768x256x0x64x1>
  %3 = migraphx.reshape %2 {dims = [2, 3, 128, 64]} : <2x3x4x32x64xf8E8M0FNU, 768x256x0x64x1> -> <2x3x128x64xf8E8M0FNU, 768x256x64x1>
  
  %4 = migraphx.quant_dot %arg0 scaled by %1, %arg1 scaled by %3
    : !migraphx.shaped<2x3x64x128xf4E2M1FN, 24576x8192x128x1>
        scaled by !migraphx.shaped<2x3x64x128xf8E8M0FNU, 768x256x4x1>,
      !migraphx.shaped<2x3x128x64xf4E2M1FN, 24576x8192x64x1>
        scaled by !migraphx.shaped<2x3x128x64xf8E8M0FNU, 768x256x64x1>
    -> !migraphx.shaped<2x3x64x64xf32, 12288x4096x64x1>
  return %4 : !migraphx.shaped<2x3x64x64xf32, 12288x4096x64x1>
}

// ============================================================================
// NEGATIVE TEST 1: quant_dot without scales - should use regular tosa.matmul
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
// NEGATIVE TEST 2: FP8 quant_dot without scales - uses regular tosa.matmul
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
