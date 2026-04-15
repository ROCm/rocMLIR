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
      migraphx.yield %sum : !migraphx.shaped<2x64x256xf16, 16384x256x1>
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
      migraphx.yield %scaled : !migraphx.shaped<1x7x7xf32, 49x7x1>
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

// Causal: decomposes to dot + greater + where(-inf) + softmax + dot
// CHECK-LABEL: func.func @decompose_causal
// CHECK: migraphx.dot
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
func.func @decompose_causal(
    %q: !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>
) -> !migraphx.shaped<1x2x4x8xf16, 64x32x8x1> {
  %0 = migraphx.attention %q, %k, %v {
  } features = causal
    : <1x2x4x8xf16, 64x32x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x4x8xf16, 64x32x8x1>
  return %0 : !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>
}

// KV-cache: decomposes to dot + greater(col, seqLen) + where + softmax + dot
// CHECK-LABEL: func.func @decompose_kvcache
// CHECK: migraphx.dot
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
func.func @decompose_kvcache(
    %q: !migraphx.shaped<1x2x1x8xf16, 16x8x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>,
    %sl: !migraphx.shaped<1x2xi32, 2x1>
) -> !migraphx.shaped<1x2x1x8xf16, 16x8x8x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<1x2xi32, 2x1>) {
    } features = kvcache
    : <1x2x1x8xf16, 16x8x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x1x8xf16, 16x8x8x1>
  return %0 : !migraphx.shaped<1x2x1x8xf16, 16x8x8x1>
}

// Sliding window: decomposes with sliding window mask then kvcache mask
// CHECK-LABEL: func.func @decompose_sliding_window
// CHECK: migraphx.literal(dense<-4>
// CHECK: migraphx.dot
// sliding window mask: lowerBound = seqLen + (-windowSize), greater(lowerBound, col)
// CHECK: migraphx.add
// CHECK: migraphx.greater
// CHECK: migraphx.where
// kvcache mask: greater(col, seqLen)
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
func.func @decompose_sliding_window(
    %q: !migraphx.shaped<1x2x1x2xf16, 4x2x2x1>,
    %k: !migraphx.shaped<1x2x2x8xf16, 32x16x8x1>,
    %v: !migraphx.shaped<1x2x8x2xf16, 32x16x2x1>,
    %sl: !migraphx.shaped<1x2xi32, 2x1>
) -> !migraphx.shaped<1x2x1x2xf16, 4x2x2x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<1x2xi32, 2x1>) {
    } features = "kvcache|sliding_window" slidingWindowSize = 4
    : <1x2x1x2xf16, 4x2x2x1>, <1x2x2x8xf16, 32x16x8x1>, <1x2x8x2xf16, 32x16x2x1>
    -> <1x2x1x2xf16, 4x2x2x1>
  return %0 : !migraphx.shaped<1x2x1x2xf16, 4x2x2x1>
}

// SplitKV=2: Q [1,2,4,8] -> reshape [1,2,1,4,8] -> broadcast [1,2,2,4,8]
//            K [1,2,8,16] -> reshape [1,2,8,2,8] -> transpose [1,2,2,8,8]
//            V [1,2,16,8] -> reshape [1,2,2,8,8]
//            dot in 5D split space, manual softmax for LSE
// CHECK-LABEL: func.func @decompose_splitkv
// CHECK: migraphx.reshape %arg0 {dims = [1, 2, 1, 4, 8]}
// CHECK: migraphx.multibroadcast {{.*}} {out_lens = [1, 2, 2, 4, 8]}
// CHECK: migraphx.reshape %arg1 {dims = [1, 2, 8, 2, 8]}
// CHECK: migraphx.transpose {{.*}} {permutation = [0, 1, 3, 2, 4]}
// CHECK-SAME: -> <1x2x2x8x8
// CHECK: migraphx.reshape %arg2 {dims = [1, 2, 2, 8, 8]}
// CHECK: migraphx.dot {{.*}} -> <1x2x2x4x8
// CHECK: migraphx.reduce_max {{.*}} -> <1x2x2x4x1
// CHECK: migraphx.exp {{.*}} -> <1x2x2x4x8
// CHECK: migraphx.reduce_sum {{.*}} -> <1x2x2x4x1
// CHECK: migraphx.dot {{.*}} -> <1x2x2x4x8
// CHECK: migraphx.reshape {{.*}} {dims = [1, 2, 2, 4]} {{.*}} -> <1x2x2x4
func.func @decompose_splitkv(
    %q: !migraphx.shaped<1x2x4x8xf32, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf32, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf32, 256x128x8x1>
) -> (!migraphx.shaped<1x2x2x4x8xf32, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>) {
  %0, %1 = migraphx.attention %q, %k, %v {
  } features = splitkv splitKV = 2
    : <1x2x4x8xf32, 64x32x8x1>, <1x2x8x16xf32, 256x128x16x1>, <1x2x16x8xf32, 256x128x8x1>
    -> <1x2x2x4x8xf32, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
  return %0, %1 : !migraphx.shaped<1x2x2x4x8xf32, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
}

// Prefix offset: causal mask with shifted boundary
// CHECK-LABEL: func.func @decompose_prefix_offset
// CHECK: migraphx.dot
// CHECK: migraphx.add
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
func.func @decompose_prefix_offset(
    %q: !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>,
    %sl: !migraphx.shaped<1x2xi32, 2x1>,
    %po: !migraphx.shaped<1xi32, 1>
) -> !migraphx.shaped<1x2x4x8xf16, 64x32x8x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<1x2xi32, 2x1>)
    prefix_offset(%po : !migraphx.shaped<1xi32, 1>) {
    } features = "kvcache|causal|prefix_offset"
    : <1x2x4x8xf16, 64x32x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x4x8xf16, 64x32x8x1>
  return %0 : !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>
}

// KV-cache + causal: both masks applied in sequence
// CHECK-LABEL: func.func @decompose_kvcache_causal
// CHECK: migraphx.dot
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
func.func @decompose_kvcache_causal(
    %q: !migraphx.shaped<1x2x1x8xf16, 16x8x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>,
    %sl: !migraphx.shaped<1x2xi32, 2x1>
) -> !migraphx.shaped<1x2x1x8xf16, 16x8x8x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<1x2xi32, 2x1>) {
    } features = "kvcache|causal"
    : <1x2x1x8xf16, 16x8x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x1x8xf16, 16x8x8x1>
  return %0 : !migraphx.shaped<1x2x1x8xf16, 16x8x8x1>
}

// KV-cache + causal + sliding window: all three masks applied
// CHECK-LABEL: func.func @decompose_kvcache_causal_sliding
// CHECK: migraphx.dot
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
func.func @decompose_kvcache_causal_sliding(
    %q: !migraphx.shaped<1x2x1x2xf16, 4x2x2x1>,
    %k: !migraphx.shaped<1x2x2x8xf16, 32x16x8x1>,
    %v: !migraphx.shaped<1x2x8x2xf16, 32x16x2x1>,
    %sl: !migraphx.shaped<1x2xi32, 2x1>
) -> !migraphx.shaped<1x2x1x2xf16, 4x2x2x1> {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<1x2xi32, 2x1>) {
    } features = "kvcache|causal|sliding_window" slidingWindowSize = 4
    : <1x2x1x2xf16, 4x2x2x1>, <1x2x2x8xf16, 32x16x8x1>, <1x2x8x2xf16, 32x16x2x1>
    -> <1x2x1x2xf16, 4x2x2x1>
  return %0 : !migraphx.shaped<1x2x1x2xf16, 4x2x2x1>
}

// Causal + preSoftmaxBody: mask applied after elementwise fusion
// CHECK-LABEL: func.func @decompose_causal_with_presoftmax
// CHECK: migraphx.dot
// CHECK: migraphx.mul
// CHECK: migraphx.add
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
func.func @decompose_causal_with_presoftmax(
    %q: !migraphx.shaped<1x7x3xf16, 21x3x1>,
    %k: !migraphx.shaped<1x3x7xf16, 21x7x1>,
    %v: !migraphx.shaped<1x7x3xf16, 21x3x1>,
    %s: !migraphx.shaped<1x7x7xf16, 49x7x1>,
    %b: !migraphx.shaped<1x7x7xf16, 49x7x1>
) -> !migraphx.shaped<1x7x3xf16, 21x3x1> {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%s, %b
      : !migraphx.shaped<1x7x7xf16, 49x7x1>,
        !migraphx.shaped<1x7x7xf16, 49x7x1>) {
    ^bb0(%qk: !migraphx.shaped<1x7x7xf16, 49x7x1>,
         %ss: !migraphx.shaped<1x7x7xf16, 49x7x1>,
         %bb: !migraphx.shaped<1x7x7xf16, 49x7x1>):
      %scaled = migraphx.mul %qk, %ss
        : <1x7x7xf16, 49x7x1>, <1x7x7xf16, 49x7x1> -> <1x7x7xf16, 49x7x1>
      %biased = migraphx.add %scaled, %bb
        : <1x7x7xf16, 49x7x1>, <1x7x7xf16, 49x7x1> -> <1x7x7xf16, 49x7x1>
      migraphx.yield %biased : !migraphx.shaped<1x7x7xf16, 49x7x1>
    } features = causal
    : <1x7x3xf16, 21x3x1>, <1x3x7xf16, 21x7x1>, <1x7x3xf16, 21x3x1>
    -> <1x7x3xf16, 21x3x1>
  return %0 : !migraphx.shaped<1x7x3xf16, 21x3x1>
}

// Causal + softmaxType: mask + precision conversion before softmax
// CHECK-LABEL: func.func @decompose_causal_softmax_f32
// CHECK: migraphx.dot
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.convert
// CHECK: migraphx.softmax
// CHECK: migraphx.convert
// CHECK: migraphx.dot
func.func @decompose_causal_softmax_f32(
    %q: !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>
) -> !migraphx.shaped<1x2x4x8xf16, 64x32x8x1> {
  %0 = migraphx.attention %q, %k, %v {
  } softmax_type = f32 features = causal
    : <1x2x4x8xf16, 64x32x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x4x8xf16, 64x32x8x1>
  return %0 : !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>
}

// preSoftmaxBody + softmaxType + causal: all three combined
// CHECK-LABEL: func.func @decompose_presoftmax_softmaxtype_causal
// CHECK: migraphx.dot
// CHECK: migraphx.mul
// CHECK: migraphx.greater
// CHECK: migraphx.where
// CHECK: migraphx.convert
// CHECK: migraphx.softmax
// CHECK: migraphx.convert
// CHECK: migraphx.dot
func.func @decompose_presoftmax_softmaxtype_causal(
    %q: !migraphx.shaped<1x7x3xf16, 21x3x1>,
    %k: !migraphx.shaped<1x3x7xf16, 21x7x1>,
    %v: !migraphx.shaped<1x7x3xf16, 21x3x1>,
    %s: !migraphx.shaped<1x7x7xf16, 49x7x1>
) -> !migraphx.shaped<1x7x3xf16, 21x3x1> {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%s : !migraphx.shaped<1x7x7xf16, 49x7x1>) {
    ^bb0(%qk: !migraphx.shaped<1x7x7xf16, 49x7x1>,
         %ss: !migraphx.shaped<1x7x7xf16, 49x7x1>):
      %scaled = migraphx.mul %qk, %ss
        : <1x7x7xf16, 49x7x1>, <1x7x7xf16, 49x7x1> -> <1x7x7xf16, 49x7x1>
      migraphx.yield %scaled : !migraphx.shaped<1x7x7xf16, 49x7x1>
    } softmax_type = f32 features = causal
    : <1x7x3xf16, 21x3x1>, <1x3x7xf16, 21x7x1>, <1x7x3xf16, 21x3x1>
    -> <1x7x3xf16, 21x3x1>
  return %0 : !migraphx.shaped<1x7x3xf16, 21x3x1>
}

// Kernel function: all attention ops are preserved (handled by MIGraphXAttentionToRock)
// CHECK-LABEL: func.func @kernel_attention_preserved
// CHECK: migraphx.attention
// CHECK: features = causal
func.func @kernel_attention_preserved(
    %q: !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>
) -> !migraphx.shaped<1x2x4x8xf16, 64x32x8x1> attributes {rock.kernel} {
  %0 = migraphx.attention %q, %k, %v {
  } features = causal
    : <1x2x4x8xf16, 64x32x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x4x8xf16, 64x32x8x1>
  return %0 : !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>
}

// SplitKV on rock.kernel function: also preserved (MIGraphXAttentionToRock handles splitkv)
// CHECK-LABEL: func.func @kernel_splitkv_preserved
// CHECK: migraphx.attention
// CHECK: features = splitkv
func.func @kernel_splitkv_preserved(
    %q: !migraphx.shaped<1x2x4x8xf32, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf32, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf32, 256x128x8x1>
) -> (!migraphx.shaped<1x2x2x4x8xf32, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>) attributes {rock.kernel} {
  %0, %1 = migraphx.attention %q, %k, %v {
  } features = splitkv splitKV = 2
    : <1x2x4x8xf32, 64x32x8x1>, <1x2x8x16xf32, 256x128x16x1>, <1x2x16x8xf32, 256x128x8x1>
    -> <1x2x2x4x8xf32, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
  return %0, %1 : !migraphx.shaped<1x2x2x4x8xf32, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
}
