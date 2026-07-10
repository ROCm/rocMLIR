// Host split-KV validity mask: per-(batch-head, query-row) layout.
//
// computeValidSplitKV() emits one validity count per batch-head for the plain
// split-KV case, but under causal / prefix-causal masking with seqLenQ > 1 the
// number of valid splits differs per query row. createMaskSplitKV() then has to
// build a per-row mask whose validity counts are laid out along the query-row
// axis ([batch, 1, seqQ, 1]) rather than the batch axis ([batch, 1, 1, 1]).
//
// This test exercises that per-row layout (causal, seqLenQ = 4, splitKV = 4)
// and also confirms the combine still runs in f32 for f16 storage.
//
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -causal=true -return_lse -split_kv 4 -seq_len_q 4 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK-LABEL: func.func @rock_attention_gpu

// Split-KV partial outputs reshaped to [batchHeads, splitKV, seqQ, headDimV],
// keeping the query-row axis (seqQ = 4) distinct from the split axis.
// CHECK: tosa.reshape %{{.+}} : (tensor<512xf16>, !tosa.shape<4>) -> tensor<1x4x4x32xf16>

// f16 storage upcast to f32 for the combine.
// CHECK: tosa.cast %{{.+}} : (tensor<1x4x4x32xf16>) -> tensor<1x4x4x32xf32>

// Per-row validity counts: laid out along the query-row axis ([batch, 1, seqQ, 1]),
// i.e. tensor<1x1x4x1xi32>, NOT the per-batch-head tensor<Nx1x1x1xi32> layout.
// CHECK: "tosa.const"() <{values = {{.+}} : tensor<1x1x4x1xi32>}>
// CHECK: tosa.greater_equal %{{.+}}, %{{.+}} : (tensor<1x4x4x1xi32>, tensor<1x4x4x1xi32>) -> tensor<1x4x4x1xi1>

// LSE re-normalization across the splitKV axis (axis = 1) in f32.
// CHECK: tosa.reduce_max %{{.+}} {axis = 1 : i32} : (tensor<1x4x4x1xf32>) -> tensor<1x1x4x1xf32>

// Final combined result cast back to f16 storage.
// CHECK: tosa.cast %{{.+}} : (tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf16>
