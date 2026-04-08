// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// RUN: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

// CHECK-LABEL: func.func @migraphx_dot
// CHECK-NEXT: migraphx.dot 
func.func @migraphx_dot(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1>  {
  %0 = migraphx.dot %arg0, %arg1 : <1x16x512xf4E2M1FN, 8192x512x1>, <1x512x16xf4E2M1FN, 8192x16x1> -> <1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}



// CHECK-LABEL: func.func @migraphx_quant_dot_scaled
// CHECK-NEXT: migraphx.quant_dot
func.func @migraphx_quant_dot_scaled(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>, %arg2: !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>, %arg3: !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1>  {
 %0 = migraphx.quant_dot
      %arg0 scaled by %arg2,
      %arg1 scaled by %arg3
    : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
      !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>,
      !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
      !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>
    -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// Checking to see if the verifier allows for broadcast
// CHECK-LABEL: func.func @migraphx_dot_no_batch_b
// CHECK-NEXT: migraphx.dot
func.func @migraphx_dot_no_batch_b(%arg0: !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>, %arg1: !migraphx.shaped<2x2xf16, 2x1>) -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1> {
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf16, 8x4x2x1>, <2x2xf16, 2x1> -> <3x2x2x2xf16, 8x4x2x1>
  return %0 : !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
}

// CHECK-LABEL: func.func @migraphx_dot_leading_ones_b_rank3
// CHECK-NEXT: migraphx.dot
func.func @migraphx_dot_leading_ones_b_rank3(%arg0: !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>, %arg1: !migraphx.shaped<1x2x2xf16, 4x2x1>) -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1> {
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf16, 8x4x2x1>, <1x2x2xf16, 4x2x1> -> <3x2x2x2xf16, 8x4x2x1>
  return %0 : !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
}

// CHECK-LABEL: func.func @migraphx_dot_leading_ones_b_rank4
// CHECK-NEXT: migraphx.dot
func.func @migraphx_dot_leading_ones_b_rank4(%arg0: !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>, %arg1: !migraphx.shaped<1x1x2x2xf16, 4x2x1x1>) -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1> {
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf16, 8x4x2x1>, <1x1x2x2xf16, 4x2x1x1> -> <3x2x2x2xf16, 8x4x2x1>
  return %0 : !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
}

// ---- Elementwise binary ops ----

// CHECK-LABEL: func.func @migraphx_add
// CHECK-NEXT: migraphx.add
func.func @migraphx_add(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.add %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_sub
// CHECK-NEXT: migraphx.sub
func.func @migraphx_sub(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.sub %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_mul
// CHECK-NEXT: migraphx.mul
func.func @migraphx_mul(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.mul %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_div
// CHECK-NEXT: migraphx.div
func.func @migraphx_div(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.div %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_pow
// CHECK-NEXT: migraphx.pow
func.func @migraphx_pow(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.pow %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_greater
// CHECK-NEXT: migraphx.greater
func.func @migraphx_greater(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.greater %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_equal
// CHECK-NEXT: migraphx.equal
func.func @migraphx_equal(%arg0: !migraphx.shaped<4x8xi8, 8x1>, %arg1: !migraphx.shaped<4x8xi8, 8x1>) -> !migraphx.shaped<4x8xi8, 8x1> {
  %0 = migraphx.equal %arg0, %arg1 : <4x8xi8, 8x1>, <4x8xi8, 8x1> -> <4x8xi8, 8x1>
  return %0 : !migraphx.shaped<4x8xi8, 8x1>
}

// CHECK-LABEL: func.func @migraphx_clip
// CHECK-NEXT: migraphx.clip
func.func @migraphx_clip(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>, %arg2: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.clip %arg0, %arg1, %arg2 : <4x8xf32, 8x1>, <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_where
// CHECK-NEXT: migraphx.where
func.func @migraphx_where(%cond: !migraphx.shaped<4x8xi8, 8x1>, %arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.where %cond, %arg0, %arg1 : <4x8xi8, 8x1>, <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- Elementwise unary ops ----

// CHECK-LABEL: func.func @migraphx_abs
// CHECK-NEXT: migraphx.abs
func.func @migraphx_abs(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.abs %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_ceil
// CHECK-NEXT: migraphx.ceil
func.func @migraphx_ceil(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.ceil %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_exp
// CHECK-NEXT: migraphx.exp
func.func @migraphx_exp(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.exp %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_neg
// CHECK-NEXT: migraphx.neg
func.func @migraphx_neg(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.neg %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_recip
// CHECK-NEXT: migraphx.recip
func.func @migraphx_recip(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.recip %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_relu
// CHECK-NEXT: migraphx.relu
func.func @migraphx_relu(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.relu %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_sigmoid
// CHECK-NEXT: migraphx.sigmoid
func.func @migraphx_sigmoid(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.sigmoid %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_sqrt
// CHECK-NEXT: migraphx.sqrt
func.func @migraphx_sqrt(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.sqrt %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_tanh
// CHECK-NEXT: migraphx.tanh
func.func @migraphx_tanh(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.tanh %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// CHECK-LABEL: func.func @migraphx_convert
// CHECK-NEXT: migraphx.convert
func.func @migraphx_convert(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf16, 8x1> {
  %0 = migraphx.convert %arg0 : <4x8xf32, 8x1> to <4x8xf16, 8x1>
  return %0 : !migraphx.shaped<4x8xf16, 8x1>
}
