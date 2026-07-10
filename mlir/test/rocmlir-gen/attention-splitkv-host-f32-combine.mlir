// Host split-KV combine reference for narrow storage types.
//
// computeFinalAttentionStage() reduces the per-split partial outputs and LSE
// back into the final attention output. For f16/bf16 storage the reduction is
// numerically sensitive (LSE re-normalization + weighted sum across splits),
// so the combine must be performed in f32: the partial outputs and LSE are
// upcast to f32, every reduction/exp/reciprocal runs in f32, and only the
// final result is cast back to the storage type. This guards against the
// precision loss / NaNs that an in-f16 combine could introduce.
//
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -current_seq_len=33 -return_lse -split_kv 8 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f16 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK-LABEL: func.func @rock_attention_gpu

// Reshape the kernel's split-KV partial outputs / LSE into
// [batchHeads, splitKV, seqQ, headDimV] (here [4, 8, 1, 32]).
// CHECK: tosa.reshape %{{.+}} : (tensor<1024xf16>, !tosa.shape<4>) -> tensor<4x8x1x32xf16>
// CHECK: tosa.reshape %{{.+}} : (tensor<32xf16>, !tosa.shape<4>) -> tensor<4x8x1x1xf16>

// f16 storage is upcast to f32 before the combine.
// CHECK: tosa.cast %{{.+}} : (tensor<4x8x1x32xf16>) -> tensor<4x8x1x32xf32>
// CHECK: tosa.cast %{{.+}} : (tensor<4x8x1x1xf16>) -> tensor<4x8x1x1xf32>

// LSE re-normalization across the splitKV axis (axis = 1), entirely in f32.
// CHECK: %[[mx:.+]] = tosa.reduce_max %{{.+}} {axis = 1 : i32} : (tensor<4x8x1x1xf32>) -> tensor<4x1x1x1xf32>
// CHECK: tosa.sub %{{.+}}, %[[mx]] : (tensor<4x8x1x1xf32>, tensor<4x1x1x1xf32>) -> tensor<4x8x1x1xf32>
// CHECK: tosa.exp %{{.+}} : (tensor<4x8x1x1xf32>) -> tensor<4x8x1x1xf32>
// CHECK: tosa.reduce_sum %{{.+}} {axis = 1 : i32} : (tensor<4x8x1x1xf32>) -> tensor<4x1x1x1xf32>
// CHECK: tosa.reciprocal %{{.+}} : (tensor<4x1x1x1xf32>) -> tensor<4x1x1x1xf32>

// Weighted sum of the partial outputs across splits, still in f32.
// CHECK: tosa.reduce_sum %{{.+}} {axis = 1 : i32} : (tensor<4x8x1x32xf32>) -> tensor<4x1x1x32xf32>

// Only the final combined result is cast back to the f16 storage type.
// CHECK: tosa.cast %{{.+}} : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32xf16>
