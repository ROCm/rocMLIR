// RUN: rocmlir-opt --migraphx-transform %s | FileCheck %s

// CHECK-LABEL: func.func @basic_decompose
// CHECK: migraphx.dot
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
// CHECK-NOT: migraphx.attention
func.func @basic_decompose(
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

// CHECK-LABEL: func.func @decompose_with_body
// CHECK: migraphx.dot
// CHECK: migraphx.add
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
// CHECK-NOT: migraphx.attention
func.func @decompose_with_body(
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
        -> <2x64x256xf16, 16384x256x1>
      migraphx.yield
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @decompose_with_softmax_type
// CHECK: migraphx.dot
// CHECK: migraphx.convert
// CHECK: migraphx.softmax
// CHECK: migraphx.convert
// CHECK: migraphx.dot
// CHECK-NOT: migraphx.attention
func.func @decompose_with_softmax_type(
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

// CHECK-LABEL: func.func @decompose_with_lse
// CHECK: migraphx.dot
// CHECK: migraphx.reduce_max
// CHECK: migraphx.sub
// CHECK: migraphx.exp
// CHECK: migraphx.reduce_sum
// CHECK: migraphx.recip
// CHECK: migraphx.mul
// CHECK: migraphx.log
// CHECK: migraphx.add
// CHECK: migraphx.dot
// CHECK: migraphx.reshape
// CHECK-NOT: migraphx.attention
// CHECK-NOT: migraphx.softmax
func.func @decompose_with_lse(
    %q: !migraphx.shaped<2x64x64xf32, 4096x64x1>,
    %k: !migraphx.shaped<2x64x64xf32, 4096x64x1>,
    %v: !migraphx.shaped<2x64x64xf32, 4096x64x1>
) -> (!migraphx.shaped<2x64x64xf32, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>) {
  %0, %1 = migraphx.attention %q, %k, %v {
  }
    : <2x64x64xf32, 4096x64x1>, <2x64x64xf32, 4096x64x1>, <2x64x64xf32, 4096x64x1>
    -> !migraphx.shaped<2x64x64xf32, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf32, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
}

// CHECK-LABEL: func.func @decompose_with_lse_and_body
// CHECK: migraphx.dot
// CHECK: migraphx.mul
// CHECK: migraphx.reduce_max
// CHECK: migraphx.sub
// CHECK: migraphx.exp
// CHECK: migraphx.reduce_sum
// CHECK: migraphx.recip
// CHECK: migraphx.mul
// CHECK: migraphx.log
// CHECK: migraphx.add
// CHECK: migraphx.dot
// CHECK-NOT: migraphx.attention
func.func @decompose_with_lse_and_body(
    %q: !migraphx.shaped<1x7x3xf32, 21x3x1>,
    %k: !migraphx.shaped<1x3x7xf32, 21x7x1>,
    %v: !migraphx.shaped<1x7x3xf32, 21x3x1>,
    %scale: !migraphx.shaped<1x7x7xf32, 49x7x1>
) -> (!migraphx.shaped<1x7x3xf32, 21x3x1>, !migraphx.shaped<1x7xf32, 7x1>) {
  %0, %1 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%scale : !migraphx.shaped<1x7x7xf32, 49x7x1>) {
    ^bb0(%qk: !migraphx.shaped<1x7x7xf32, 49x7x1>,
         %s: !migraphx.shaped<1x7x7xf32, 49x7x1>):
      %scaled = migraphx.mul %qk, %s
        : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
      migraphx.yield
    }
    : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1>, <1x7x3xf32, 21x3x1>
    -> !migraphx.shaped<1x7x3xf32, 21x3x1>, !migraphx.shaped<1x7xf32, 7x1>
  return %0, %1 : !migraphx.shaped<1x7x3xf32, 21x3x1>, !migraphx.shaped<1x7xf32, 7x1>
}

// GQA: Q has 4 heads, K/V have 2 heads. K/V should be broadcast to match Q.
// CHECK-LABEL: func.func @decompose_gqa
// CHECK: migraphx.multibroadcast
// CHECK: migraphx.reshape
// CHECK: migraphx.multibroadcast
// CHECK: migraphx.reshape
// CHECK: migraphx.dot
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
// CHECK-NOT: migraphx.attention
func.func @decompose_gqa(
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

// LSE with f16 inputs and f32 LSE output: the decomposed softmax runs in f16,
// then the LSE value is converted to f32 to match the output type.
// CHECK-LABEL: func.func @decompose_lse_type_convert
// CHECK: migraphx.dot
// CHECK: migraphx.reduce_max
// CHECK: migraphx.sub
// CHECK: migraphx.exp
// CHECK: migraphx.reduce_sum
// CHECK: migraphx.recip
// CHECK: migraphx.mul
// CHECK: migraphx.log
// CHECK: migraphx.add
// CHECK: migraphx.dot
// CHECK: migraphx.reshape
// CHECK: migraphx.convert
// CHECK-NOT: migraphx.attention
func.func @decompose_lse_type_convert(
    %q: !migraphx.shaped<2x64x64xf16, 4096x64x1>,
    %k: !migraphx.shaped<2x64x64xf16, 4096x64x1>,
    %v: !migraphx.shaped<2x64x64xf16, 4096x64x1>
) -> (!migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>) {
  %0, %1 = migraphx.attention %q, %k, %v {
  }
    : <2x64x64xf16, 4096x64x1>, <2x64x64xf16, 4096x64x1>, <2x64x64xf16, 4096x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
}
