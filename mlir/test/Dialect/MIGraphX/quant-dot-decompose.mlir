// RUN: rocmlir-opt %s -migraphx-transform -split-input-file | FileCheck %s

// ===----------------------------------------------------------------------===//
// Valid scaled quant_dot operations that should be decomposed
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @quant_dot_with_both_scales_f8e8m0fnu
// CHECK-SAME: (%[[ARG0:.*]]: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
// CHECK-SAME:  %[[ARG1:.*]]: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
// CHECK-SAME:  %[[ARG2:.*]]: !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>,
// CHECK-SAME:  %[[ARG3:.*]]: !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>)
func.func @quant_dot_with_both_scales_f8e8m0fnu(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>,
  %arg3: !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> {
  // CHECK-DAG: %[[CVT_A:.*]] = migraphx.convert %[[ARG0]] : <1x16x512xf4E2M1FN, 8192x512x1> to <1x16x512xf32, 8192x512x1>
  // CHECK-DAG: %[[CVT_B:.*]] = migraphx.convert %[[ARG1]] : <1x512x16xf4E2M1FN, 8192x16x1> to <1x512x16xf32, 8192x16x1>
  // CHECK-DAG: %[[CVT_SCALE_A:.*]] = migraphx.convert %[[ARG2]] : <1x16x512xf8E8M0FNU, 8192x512x1> to <1x16x512xf32, 8192x512x1>
  // CHECK-DAG: %[[CVT_SCALE_B:.*]] = migraphx.convert %[[ARG3]] : <1x512x16xf8E8M0FNU, 8192x16x1> to <1x512x16xf32, 8192x16x1>
  // CHECK-DAG: %[[MUL_A:.*]] = migraphx.mul %[[CVT_A]], %[[CVT_SCALE_A]] : <1x16x512xf32, 8192x512x1>, <1x16x512xf32, 8192x512x1> -> <1x16x512xf32, 8192x512x1>
  // CHECK-DAG: %[[MUL_B:.*]] = migraphx.mul %[[CVT_B]], %[[CVT_SCALE_B]] : <1x512x16xf32, 8192x16x1>, <1x512x16xf32, 8192x16x1> -> <1x512x16xf32, 8192x16x1>
  // CHECK: %[[DOT:.*]] = migraphx.dot %[[MUL_A]], %[[MUL_B]] {perf_config = "v3:64,64,16,32,32,32,4,1,2,1,1"} : <1x16x512xf32, 8192x512x1>, <1x512x16xf32, 8192x16x1> -> <1x16x16xf32, 256x16x1>
  // CHECK: return %[[DOT]]
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3 {perf_config = "v3:64,64,16,32,32,32,4,1,2,1,1"}
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
       !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>,
       !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
       !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>
     -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// CHECK-LABEL: func.func @quant_dot_with_both_scales_f32
// CHECK-SAME: (%[[ARG0:.*]]: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
// CHECK-SAME:  %[[ARG1:.*]]: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
// CHECK-SAME:  %[[ARG2:.*]]: !migraphx.shaped<1x16x512xf32, 8192x512x1>,
// CHECK-SAME:  %[[ARG3:.*]]: !migraphx.shaped<1x512x16xf32, 8192x16x1>)
func.func @quant_dot_with_both_scales_f32(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x16x512xf32, 8192x512x1>,
  %arg3: !migraphx.shaped<1x512x16xf32, 8192x16x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> {
  // CHECK-DAG: %[[CVT_A:.*]] = migraphx.convert %[[ARG0]] : <1x16x512xf4E2M1FN, 8192x512x1> to <1x16x512xf32, 8192x512x1>
  // CHECK-DAG: %[[CVT_B:.*]] = migraphx.convert %[[ARG1]] : <1x512x16xf4E2M1FN, 8192x16x1> to <1x512x16xf32, 8192x16x1>
  // CHECK-DAG: %[[MUL_A:.*]] = migraphx.mul %[[CVT_A]], %[[ARG2]] : <1x16x512xf32, 8192x512x1>, <1x16x512xf32, 8192x512x1> -> <1x16x512xf32, 8192x512x1>
  // CHECK-DAG: %[[MUL_B:.*]] = migraphx.mul %[[CVT_B]], %[[ARG3]] : <1x512x16xf32, 8192x16x1>, <1x512x16xf32, 8192x16x1> -> <1x512x16xf32, 8192x16x1>
  // CHECK: %[[DOT:.*]] = migraphx.dot %[[MUL_A]], %[[MUL_B]] : <1x16x512xf32, 8192x512x1>, <1x512x16xf32, 8192x16x1> -> <1x16x16xf32, 256x16x1>
  // CHECK: return %[[DOT]]
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
       !migraphx.shaped<1x16x512xf32, 8192x512x1>,
       !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
       !migraphx.shaped<1x512x16xf32, 8192x16x1>
     -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// Test that quant_dot without scales is NOT decomposed (should remain as-is)
// CHECK-LABEL: func.func @quant_dot_no_scales
// CHECK-SAME: (%[[ARG0:.*]]: !migraphx.shaped<1x16x512xf8E4M3FN, 8192x512x1>, 
// CHECK-SAME:  %[[ARG1:.*]]: !migraphx.shaped<1x512x16xf8E4M3FN, 8192x16x1>)
func.func @quant_dot_no_scales(
  %arg0: !migraphx.shaped<1x16x512xf8E4M3FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf8E4M3FN, 8192x16x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> {
  // CHECK: %[[QUANT_DOT:.*]] = migraphx.quant_dot
  // CHECK-NOT: migraphx.convert
  // CHECK-NOT: migraphx.mul
  // CHECK: return %[[QUANT_DOT]]
  %0 = migraphx.quant_dot
       %arg0,
       %arg1
     : <1x16x512xf8E4M3FN, 8192x512x1>,
       <1x512x16xf8E4M3FN, 8192x16x1>
     -> <1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// Test with different batch dimensions
// CHECK-LABEL: func.func @quant_dot_batched
// CHECK-SAME: (%[[ARG0:.*]]: !migraphx.shaped<8x128x256xf4E2M1FN, 32768x256x1>, 
// CHECK-SAME:  %[[ARG1:.*]]: !migraphx.shaped<8x256x128xf4E2M1FN, 32768x128x1>,
// CHECK-SAME:  %[[ARG2:.*]]: !migraphx.shaped<8x128x256xf32, 32768x256x1>,
// CHECK-SAME:  %[[ARG3:.*]]: !migraphx.shaped<8x256x128xf32, 32768x128x1>)
func.func @quant_dot_batched(
  %arg0: !migraphx.shaped<8x128x256xf4E2M1FN, 32768x256x1>, 
  %arg1: !migraphx.shaped<8x256x128xf4E2M1FN, 32768x128x1>,
  %arg2: !migraphx.shaped<8x128x256xf32, 32768x256x1>,
  %arg3: !migraphx.shaped<8x256x128xf32, 32768x128x1>
) -> !migraphx.shaped<8x128x128xf32, 16384x128x1> {
  // CHECK-DAG: %[[CVT_A:.*]] = migraphx.convert %[[ARG0]] : <8x128x256xf4E2M1FN, 32768x256x1> to <8x128x256xf32, 32768x256x1>
  // CHECK-DAG: %[[CVT_B:.*]] = migraphx.convert %[[ARG1]] : <8x256x128xf4E2M1FN, 32768x128x1> to <8x256x128xf32, 32768x128x1>
  // CHECK-DAG: %[[MUL_A:.*]] = migraphx.mul %[[CVT_A]], %[[ARG2]] : <8x128x256xf32, 32768x256x1>, <8x128x256xf32, 32768x256x1> -> <8x128x256xf32, 32768x256x1>
  // CHECK-DAG: %[[MUL_B:.*]] = migraphx.mul %[[CVT_B]], %[[ARG3]] : <8x256x128xf32, 32768x128x1>, <8x256x128xf32, 32768x128x1> -> <8x256x128xf32, 32768x128x1>
  // CHECK: %[[DOT:.*]] = migraphx.dot %[[MUL_A]], %[[MUL_B]] : <8x128x256xf32, 32768x256x1>, <8x256x128xf32, 32768x128x1> -> <8x128x128xf32, 16384x128x1>
  // CHECK: return %[[DOT]]
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<8x128x256xf4E2M1FN, 32768x256x1> scaled by
       !migraphx.shaped<8x128x256xf32, 32768x256x1>,
       !migraphx.shaped<8x256x128xf4E2M1FN, 32768x128x1> scaled by
       !migraphx.shaped<8x256x128xf32, 32768x128x1>
     -> !migraphx.shaped<8x128x128xf32, 16384x128x1>
  return %0 : !migraphx.shaped<8x128x128xf32, 16384x128x1>
}

// -----

// Test with non-standard strides to ensure stride information is preserved
// CHECK-LABEL: func.func @quant_dot_non_standard_strides
// CHECK-SAME: (%[[ARG0:.*]]: !migraphx.shaped<4x64x128xf4E2M1FN, 16384x128x1>, 
// CHECK-SAME:  %[[ARG1:.*]]: !migraphx.shaped<4x128x64xf4E2M1FN, 16384x64x1>,
// CHECK-SAME:  %[[ARG2:.*]]: !migraphx.shaped<4x64x128xf8E8M0FNU, 16384x128x1>,
// CHECK-SAME:  %[[ARG3:.*]]: !migraphx.shaped<4x128x64xf8E8M0FNU, 16384x64x1>)
func.func @quant_dot_non_standard_strides(
  %arg0: !migraphx.shaped<4x64x128xf4E2M1FN, 16384x128x1>, 
  %arg1: !migraphx.shaped<4x128x64xf4E2M1FN, 16384x64x1>,
  %arg2: !migraphx.shaped<4x64x128xf8E8M0FNU, 16384x128x1>,
  %arg3: !migraphx.shaped<4x128x64xf8E8M0FNU, 16384x64x1>
) -> !migraphx.shaped<4x64x64xf32, 4096x64x1> {
  // CHECK-DAG: %[[CVT_A:.*]] = migraphx.convert %[[ARG0]] : <4x64x128xf4E2M1FN, 16384x128x1> to <4x64x128xf32, 16384x128x1>
  // CHECK-DAG: %[[CVT_B:.*]] = migraphx.convert %[[ARG1]] : <4x128x64xf4E2M1FN, 16384x64x1> to <4x128x64xf32, 16384x64x1>
  // CHECK-DAG: %[[CVT_SCALE_A:.*]] = migraphx.convert %[[ARG2]] : <4x64x128xf8E8M0FNU, 16384x128x1> to <4x64x128xf32, 16384x128x1>
  // CHECK-DAG: %[[CVT_SCALE_B:.*]] = migraphx.convert %[[ARG3]] : <4x128x64xf8E8M0FNU, 16384x64x1> to <4x128x64xf32, 16384x64x1>
  // CHECK-DAG: %[[MUL_A:.*]] = migraphx.mul %[[CVT_A]], %[[CVT_SCALE_A]] : <4x64x128xf32, 16384x128x1>, <4x64x128xf32, 16384x128x1> -> <4x64x128xf32, 16384x128x1>
  // CHECK-DAG: %[[MUL_B:.*]] = migraphx.mul %[[CVT_B]], %[[CVT_SCALE_B]] : <4x128x64xf32, 16384x64x1>, <4x128x64xf32, 16384x64x1> -> <4x128x64xf32, 16384x64x1>
  // CHECK: %[[DOT:.*]] = migraphx.dot %[[MUL_A]], %[[MUL_B]] : <4x64x128xf32, 16384x128x1>, <4x128x64xf32, 16384x64x1> -> <4x64x64xf32, 4096x64x1>
  // CHECK: return %[[DOT]]
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<4x64x128xf4E2M1FN, 16384x128x1> scaled by
       !migraphx.shaped<4x64x128xf8E8M0FNU, 16384x128x1>,
       !migraphx.shaped<4x128x64xf4E2M1FN, 16384x64x1> scaled by
       !migraphx.shaped<4x128x64xf8E8M0FNU, 16384x64x1>
     -> !migraphx.shaped<4x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<4x64x64xf32, 4096x64x1>
}

// Kernel functions shouldn't get decomposed
// CHECK-LABEL: func.func @quant_dot_with_both_scales_f8e8m0fnu_kernel
// CHECK-NOT: migraphx.convert
// CHECK-NOT: migraphx.dot
// CHECK: migraphx.quant_dot
func.func @quant_dot_with_both_scales_f8e8m0fnu_kernel(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>,
  %arg3: !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> attributes {kernel} {
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3 {perf_config = "v3:64,64,16,32,32,32,4,1,2,1,1"}
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
       !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>,
       !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
       !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>
     -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}
