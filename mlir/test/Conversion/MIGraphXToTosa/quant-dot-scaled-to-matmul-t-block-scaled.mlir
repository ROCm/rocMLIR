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
