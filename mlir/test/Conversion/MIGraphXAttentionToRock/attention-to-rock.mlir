// RUN: rocmlir-opt --migraphx-attention-to-rock %s | FileCheck %s

// CHECK-LABEL: func.func @basic_attention
// CHECK: rock.attention
// CHECK: softmax(qk)
// CHECK: numHeadsQ = 1
// CHECK-SAME: storeMethod = #rock<StoreMethod set>
func.func @basic_attention(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @attention_with_softmax_type
// CHECK: rock.attention
// CHECK: softmaxType = f32
func.func @attention_with_softmax_type(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v {
  } softmax_type = f32
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @attention_with_pre_softmax_body
// CHECK: rock.attention
// CHECK: elementwise
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: rock.yield
func.func @attention_with_pre_softmax_body(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %bias: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
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

// CHECK-LABEL: func.func @attention_with_lse
// CHECK: %[[RESULT:.+]], %[[LSE:.+]] = rock.attention
// CHECK: lse = %{{.+}} : tensor<2x64xf32>
// CHECK: -> tensor<2x64x64xf16>, tensor<2x64xf32>
// CHECK: migraphx.mlir.as.underlying.shape %[[RESULT]]
// CHECK: migraphx.mlir.as.underlying.shape %[[LSE]]
func.func @attention_with_lse(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>) attributes {rock.kernel, arch = ""} {
  %0, %1 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
}

// CHECK-LABEL: func.func @attention_with_perf_config
// CHECK: rock.attention
// CHECK: perf_config = "attn:v2:128,128,128,2,64,64,8,4,1,2,1"
func.func @attention_with_perf_config(
    %q: !migraphx.shaped<2x64x64xf32, 4096x64x1>,
    %k: !migraphx.shaped<2x64x64xf32, 4096x64x1>,
    %v: !migraphx.shaped<2x64x64xf32, 4096x64x1>
) -> !migraphx.shaped<2x64x64xf32, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v {
  } {perf_config = "attn:v2:128,128,128,2,64,64,8,4,1,2,1"}
    : <2x64x64xf32, 4096x64x1>, <2x64x64xf32, 4096x64x1>, <2x64x64xf32, 4096x64x1>
    -> !migraphx.shaped<2x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf32, 4096x64x1>
}

// CHECK-LABEL: func.func @attention_gqa_4d
// CHECK: rock.attention
// CHECK: numHeadsKV = 2
// CHECK-SAME: numHeadsQ = 4
func.func @attention_gqa_4d(
    %q: !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>,
    %k: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>,
    %v: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>
) -> !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x4x32x32xf32, 4096x1024x32x1>, <2x2x32x32xf32, 2048x1024x32x1>, <2x2x32x32xf32, 2048x1024x32x1>
    -> !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>
  return %0 : !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>
}

// GQA with 3D tensors: Q was reshaped from [2, 4, 32, 32] to [8, 32, 32],
// K/V were reshaped from [2, 2, 32, 32] to [4, 32, 32].
// The pass looks through migraphx.reshape to infer numHeads.
// CHECK-LABEL: func.func @attention_gqa_3d
// CHECK: rock.attention
// CHECK: numHeadsKV = 2
// CHECK-SAME: numHeadsQ = 4
func.func @attention_gqa_3d(
    %q_orig: !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>,
    %k_orig: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>,
    %v_orig: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>
) -> !migraphx.shaped<8x32x32xf32, 1024x32x1> attributes {rock.kernel, arch = ""} {
  %q = migraphx.reshape %q_orig {dims = [8, 32, 32]}
    : <2x4x32x32xf32, 4096x1024x32x1> -> <8x32x32xf32, 1024x32x1>
  %k = migraphx.reshape %k_orig {dims = [4, 32, 32]}
    : <2x2x32x32xf32, 2048x1024x32x1> -> <4x32x32xf32, 1024x32x1>
  %v = migraphx.reshape %v_orig {dims = [4, 32, 32]}
    : <2x2x32x32xf32, 2048x1024x32x1> -> <4x32x32xf32, 1024x32x1>
  %0 = migraphx.attention %q, %k, %v {
  }
    : <8x32x32xf32, 1024x32x1>, <4x32x32xf32, 1024x32x1>, <4x32x32xf32, 1024x32x1>
    -> !migraphx.shaped<8x32x32xf32, 1024x32x1>
  return %0 : !migraphx.shaped<8x32x32xf32, 1024x32x1>
}

// CHECK-LABEL: func.func @attention_pre_softmax_mul
// CHECK: rock.attention
// CHECK: elementwise
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: rock.yield
func.func @attention_pre_softmax_mul(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %scale: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%scale : !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %s: !migraphx.shaped<2x64x256xf16, 16384x256x1>):
      %prod = migraphx.mul %qk, %s
        : <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> <2x64x256xf16, 16384x256x1>
      migraphx.yield %prod : !migraphx.shaped<2x64x256xf16, 16384x256x1>
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// Chained ops: scale then add bias in preSoftmaxBody, fused into single generic
// CHECK-LABEL: func.func @attention_pre_softmax_mul_add
// CHECK: rock.attention
// CHECK: elementwise
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: rock.yield
func.func @attention_pre_softmax_mul_add(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %scale: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
    %bias: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%scale, %bias
      : !migraphx.shaped<2x64x256xf16, 16384x256x1>,
        !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %s: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %b: !migraphx.shaped<2x64x256xf16, 16384x256x1>):
      %scaled = migraphx.mul %qk, %s
        : <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> <2x64x256xf16, 16384x256x1>
      %biased = migraphx.add %scaled, %b
        : <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> <2x64x256xf16, 16384x256x1>
      migraphx.yield %biased : !migraphx.shaped<2x64x256xf16, 16384x256x1>
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// 4D GQA: verifies collapse_shape and expand_shape for 4D -> 3D -> 4D
// CHECK-LABEL: func.func @attention_gqa_4d_collapse
// CHECK: tensor.collapse_shape
// CHECK: rock.attention
// CHECK: tensor.expand_shape
func.func @attention_gqa_4d_collapse(
    %q: !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>,
    %k: !migraphx.shaped<2x2x64x32xf16, 4096x2048x32x1>,
    %v: !migraphx.shaped<2x2x32x64xf16, 4096x2048x64x1>
) -> !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x4x32x64xf16, 8192x2048x64x1>, <2x2x64x32xf16, 4096x2048x32x1>, <2x2x32x64xf16, 4096x2048x64x1>
    -> !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>
  return %0 : !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>
}

// CHECK-LABEL: func.func @non_kernel_preserved
// CHECK: migraphx.attention
// CHECK-NOT: rock.attention
func.func @non_kernel_preserved(
    %q: !migraphx.shaped<2x64x64xf32, 4096x64x1>,
    %k: !migraphx.shaped<2x64x64xf32, 4096x64x1>,
    %v: !migraphx.shaped<2x64x64xf32, 4096x64x1>
) -> !migraphx.shaped<2x64x64xf32, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x64x64xf32, 4096x64x1>, <2x64x64xf32, 4096x64x1>, <2x64x64xf32, 4096x64x1>
    -> !migraphx.shaped<2x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf32, 4096x64x1>
}

// CHECK-LABEL: func.func @attention_causal
// CHECK: rock.attention
// CHECK: causal
func.func @attention_causal(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v {
  } features = causal
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @attention_kvcache
// CHECK: rock.attention
// CHECK: currentSeqLen
func.func @attention_kvcache(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = kvcache
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @attention_sliding_window
// CHECK: rock.attention
// CHECK: slidingWindowSize = 64
func.func @attention_sliding_window(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|sliding_window" slidingWindowSize = 64
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @attention_prefix_offset
// CHECK: rock.attention
// CHECK: prefixOffset
// CHECK: causal
func.func @attention_prefix_offset(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>,
    %po: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>)
    prefix_offset(%po : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|causal|prefix_offset"
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// CHECK-LABEL: func.func @attention_splitkv
// CHECK: rock.attention
// CHECK: splitKV = 2
func.func @attention_splitkv(
    %q: !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>
) -> (!migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>) attributes {rock.kernel, arch = ""} {
  %0, %1 = migraphx.attention %q, %k, %v {
  } softmax_type = f32 features = splitkv splitKV = 2
    : <1x2x4x8xf16, 64x32x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
  return %0, %1 : !migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
}

// causal + softmaxType
// CHECK-LABEL: func.func @attention_causal_softmax_f32
// CHECK: rock.attention
// CHECK: causal
// CHECK: softmaxType = f32
func.func @attention_causal_softmax_f32(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v {
  } softmax_type = f32 features = causal
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> <2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// kvcache + LSE
// CHECK-LABEL: func.func @attention_kvcache_lse
// CHECK: rock.attention
// CHECK: currentSeqLen
// CHECK-NOT: causal
func.func @attention_kvcache_lse(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> (!migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>) attributes {rock.kernel, arch = ""} {
  %0, %1 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } softmax_type = f32 features = kvcache
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> <2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
}

// splitKV + preSoftmaxBody: inputs must have split-space shapes (5D)
// CHECK-LABEL: func.func @attention_splitkv_presoftmax
// CHECK: rock.attention
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: splitKV = 2
func.func @attention_splitkv_presoftmax(
    %q: !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>,
    %s: !migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>
) -> (!migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>) attributes {rock.kernel, arch = ""} {
  %0, %1 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%s : !migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>) {
    ^bb0(%qk: !migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>,
         %ss: !migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>):
      %scaled = migraphx.mul %qk, %ss
        : <1x2x2x4x8xf16, 128x64x32x8x1>, <1x2x2x4x8xf16, 128x64x32x8x1> -> <1x2x2x4x8xf16, 128x64x32x8x1>
      migraphx.yield %scaled : !migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>
    } softmax_type = f32 features = splitkv splitKV = 2
    : <1x2x4x8xf16, 64x32x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
  return %0, %1 : !migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
}
