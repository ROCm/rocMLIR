// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// RUN: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

// ---- migraphx.add ----

// CHECK-LABEL: func.func @migraphx_add
// CHECK-NEXT: migraphx.add
func.func @migraphx_add(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.add %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.sub ----

// CHECK-LABEL: func.func @migraphx_sub
// CHECK-NEXT: migraphx.sub
func.func @migraphx_sub(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.sub %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.mul ----

// CHECK-LABEL: func.func @migraphx_mul
// CHECK-NEXT: migraphx.mul
func.func @migraphx_mul(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.mul %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.div ----

// CHECK-LABEL: func.func @migraphx_div
// CHECK-NEXT: migraphx.div
func.func @migraphx_div(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.div %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.pow ----

// CHECK-LABEL: func.func @migraphx_pow
// CHECK-NEXT: migraphx.pow
func.func @migraphx_pow(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.pow %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.greater ----

// CHECK-LABEL: func.func @migraphx_greater
// CHECK-NEXT: migraphx.greater
func.func @migraphx_greater(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.greater %arg0, %arg1 : <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.equal ----

// CHECK-LABEL: func.func @migraphx_equal
// CHECK-NEXT: migraphx.equal
func.func @migraphx_equal(%arg0: !migraphx.shaped<4x8xi8, 8x1>, %arg1: !migraphx.shaped<4x8xi8, 8x1>) -> !migraphx.shaped<4x8xi8, 8x1> {
  %0 = migraphx.equal %arg0, %arg1 : <4x8xi8, 8x1>, <4x8xi8, 8x1> -> <4x8xi8, 8x1>
  return %0 : !migraphx.shaped<4x8xi8, 8x1>
}

// ---- migraphx.clip ----

// CHECK-LABEL: func.func @migraphx_clip
// CHECK-NEXT: migraphx.clip
func.func @migraphx_clip(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>, %arg2: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.clip %arg0, %arg1, %arg2 : <4x8xf32, 8x1>, <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.where ----

// CHECK-LABEL: func.func @migraphx_where
// CHECK-NEXT: migraphx.where
func.func @migraphx_where(%cond: !migraphx.shaped<4x8xi8, 8x1>, %arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.where %cond, %arg0, %arg1 : <4x8xi8, 8x1>, <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.convert ----

// CHECK-LABEL: func.func @migraphx_convert
// CHECK-NEXT: migraphx.convert
func.func @migraphx_convert(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf16, 8x1> {
  %0 = migraphx.convert %arg0 : <4x8xf32, 8x1> to <4x8xf16, 8x1>
  return %0 : !migraphx.shaped<4x8xf16, 8x1>
}

// ---- migraphx.abs ----

// CHECK-LABEL: func.func @migraphx_abs
// CHECK-NEXT: migraphx.abs
func.func @migraphx_abs(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.abs %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.ceil ----

// CHECK-LABEL: func.func @migraphx_ceil
// CHECK-NEXT: migraphx.ceil
func.func @migraphx_ceil(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.ceil %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.exp ----

// CHECK-LABEL: func.func @migraphx_exp
// CHECK-NEXT: migraphx.exp
func.func @migraphx_exp(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.exp %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.neg ----

// CHECK-LABEL: func.func @migraphx_neg
// CHECK-NEXT: migraphx.neg
func.func @migraphx_neg(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.neg %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.recip ----

// CHECK-LABEL: func.func @migraphx_recip
// CHECK-NEXT: migraphx.recip
func.func @migraphx_recip(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.recip %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.relu ----

// CHECK-LABEL: func.func @migraphx_relu
// CHECK-NEXT: migraphx.relu
func.func @migraphx_relu(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.relu %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.sigmoid ----

// CHECK-LABEL: func.func @migraphx_sigmoid
// CHECK-NEXT: migraphx.sigmoid
func.func @migraphx_sigmoid(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.sigmoid %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.sqrt ----

// CHECK-LABEL: func.func @migraphx_sqrt
// CHECK-NEXT: migraphx.sqrt
func.func @migraphx_sqrt(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.sqrt %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.tanh ----

// CHECK-LABEL: func.func @migraphx_tanh
// CHECK-NEXT: migraphx.tanh
func.func @migraphx_tanh(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  %0 = migraphx.tanh %arg0 : <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// ---- migraphx.dot ----

// CHECK-LABEL: func.func @migraphx_dot
// CHECK-NEXT: migraphx.dot
func.func @migraphx_dot(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1>  {
  %0 = migraphx.dot %arg0, %arg1 : <1x16x512xf4E2M1FN, 8192x512x1>, <1x512x16xf4E2M1FN, 8192x16x1> -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// CHECK-LABEL: func.func @migraphx_dot_no_batch_b
// CHECK-NEXT: migraphx.dot
func.func @migraphx_dot_no_batch_b(%arg0: !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>, %arg1: !migraphx.shaped<2x2xf16, 2x1>) -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1> {
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf16, 8x4x2x1>, <2x2xf16, 2x1> -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
  return %0 : !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
}

// CHECK-LABEL: func.func @migraphx_dot_leading_ones_b_rank3
// CHECK-NEXT: migraphx.dot
func.func @migraphx_dot_leading_ones_b_rank3(%arg0: !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>, %arg1: !migraphx.shaped<1x2x2xf16, 4x2x1>) -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1> {
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf16, 8x4x2x1>, <1x2x2xf16, 4x2x1> -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
  return %0 : !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
}

// CHECK-LABEL: func.func @migraphx_dot_leading_ones_b_rank4
// CHECK-NEXT: migraphx.dot
func.func @migraphx_dot_leading_ones_b_rank4(%arg0: !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>, %arg1: !migraphx.shaped<1x1x2x2xf16, 4x2x1x1>) -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1> {
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf16, 8x4x2x1>, <1x1x2x2xf16, 4x2x1x1> -> !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
  return %0 : !migraphx.shaped<3x2x2x2xf16, 8x4x2x1>
}

// ---- migraphx.quant_dot ----

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

// ---- migraphx.attention ----


// CHECK-LABEL: func.func @migraphx_attention_basic
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_basic(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_with_lse
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_with_lse(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>) {
  %0, %1 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_with_softmax_type
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_with_softmax_type(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v {
  } softmax_type = f32
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_with_pre_softmax
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_with_pre_softmax(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %bias: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%bias : !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %b: !migraphx.shaped<2x64x256xf16, 16384x256x1>):
      %sum = migraphx.add %qk, %b
        : <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> !migraphx.shaped<2x64x256xf16, 16384x256x1>
      migraphx.yield
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_pre_softmax_add_bias
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_pre_softmax_add_bias(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %bias: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%bias : !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %b: !migraphx.shaped<2x64x256xf16, 16384x256x1>):
      %sum = migraphx.add %qk, %b
        : <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> !migraphx.shaped<2x64x256xf16, 16384x256x1>
      migraphx.yield
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_pre_softmax_scale
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_pre_softmax_scale(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %scale: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%scale : !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %s: !migraphx.shaped<2x64x256xf16, 16384x256x1>):
      %prod = migraphx.mul %qk, %s
        : <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> !migraphx.shaped<2x64x256xf16, 16384x256x1>
      migraphx.yield
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_pre_softmax_scale_and_mask
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_pre_softmax_scale_and_mask(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %scale: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
    %mask: !migraphx.shaped<2x64x256xsi8, 16384x256x1>,
    %fill: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%scale, %mask, %fill
      : !migraphx.shaped<2x64x256xf16, 16384x256x1>,
        !migraphx.shaped<2x64x256xsi8, 16384x256x1>,
        !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %s: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %m: !migraphx.shaped<2x64x256xsi8, 16384x256x1>,
         %f: !migraphx.shaped<2x64x256xf16, 16384x256x1>):
      %scaled = migraphx.mul %qk, %s
        : <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> !migraphx.shaped<2x64x256xf16, 16384x256x1>
      %masked = migraphx.where %m, %scaled, %f
        : <2x64x256xsi8, 16384x256x1>, <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> !migraphx.shaped<2x64x256xf16, 16384x256x1>
      migraphx.yield
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_bf16
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_bf16(
    %q: !migraphx.shaped<4x32x64xbf16, 2048x64x1>,
    %k: !migraphx.shaped<4x64x128xbf16, 8192x128x1>,
    %v: !migraphx.shaped<4x128x32xbf16, 4096x32x1>
) -> !migraphx.shaped<4x32x32xbf16, 1024x32x1> {
  %0 = migraphx.attention %q, %k, %v {
  }
    : <4x32x64xbf16, 2048x64x1>, <4x64x128xbf16, 8192x128x1>, <4x128x32xbf16, 4096x32x1>
    -> !migraphx.shaped<4x32x32xbf16, 1024x32x1>
  return %0 : !migraphx.shaped<4x32x32xbf16, 1024x32x1>
}

// CHECK-LABEL: func.func @migraphx_attention_i8_qk
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_i8_qk(
    %q: !migraphx.shaped<2x64x128xi8, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xi8, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xi8, 8192x128x1>, <2x128x256xi8, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// GQA: numHeadsQ=4 is divisible by numHeadsKV=2
// CHECK-LABEL: func.func @migraphx_attention_gqa
// CHECK-NEXT: migraphx.attention
func.func @migraphx_attention_gqa(
    %q: !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>,
    %k: !migraphx.shaped<2x2x64x32xf16, 4096x2048x32x1>,
    %v: !migraphx.shaped<2x2x32x64xf16, 4096x2048x64x1>
) -> !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1> {
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x4x32x64xf16, 8192x2048x64x1>, <2x2x64x32xf16, 4096x2048x32x1>, <2x2x32x64xf16, 4096x2048x64x1>
    -> !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>
  return %0 : !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_causal
// CHECK: migraphx.attention
// CHECK: features = causal
func.func @migraphx_attention_causal(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v {
  } features = causal
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_kvcache
// CHECK: migraphx.attention
// CHECK: current_seq_len
// CHECK: features = kvcache
func.func @migraphx_attention_kvcache(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = kvcache
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_kvcache_causal
// CHECK: migraphx.attention
// CHECK: features = "kvcache|causal"
func.func @migraphx_attention_kvcache_causal(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|causal"
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_splitkv
// CHECK: migraphx.attention
// CHECK: features = splitkv
// CHECK: splitKV = 2
func.func @migraphx_attention_splitkv(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x2x64x64xf16, 8192x4096x64x1>, !migraphx.shaped<2x2x64xf32, 128x64x1>) {
  %0, %1 = migraphx.attention %q, %k, %v {
  } features = splitkv splitKV = 2
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x2x64x64xf16, 8192x4096x64x1>, !migraphx.shaped<2x2x64xf32, 128x64x1>
  return %0, %1 : !migraphx.shaped<2x2x64x64xf16, 8192x4096x64x1>, !migraphx.shaped<2x2x64xf32, 128x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_kvcache_causal_prefix
// CHECK: migraphx.attention
// CHECK: prefix_offset
// CHECK: features = "kvcache|causal|prefix_offset"
func.func @migraphx_attention_kvcache_causal_prefix(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>,
    %po: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>)
    prefix_offset(%po : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|causal|prefix_offset"
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @migraphx_attention_kvcache_causal_sliding_window
// CHECK: current_seq_len
// CHECK: features = "kvcache|causal|sliding_window"
// CHECK: slidingWindowSize = 64
func.func @migraphx_attention_kvcache_causal_sliding_window(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|causal|sliding_window" slidingWindowSize = 64
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

