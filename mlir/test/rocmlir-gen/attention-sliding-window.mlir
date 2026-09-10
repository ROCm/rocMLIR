// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -last_valid_kv_index=33 -sliding_window_look_back=16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=SAFE
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -g 2 -sliding_window_look_back=16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --check-prefix=DEFAULT-LAST-VALID-KV-INDEX
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -last_valid_kv_index=63 -sliding_window_look_back=63 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --check-prefix=MAX-LOOK-BACK
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -last_valid_kv_index=0 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --check-prefix=NON-SLIDING

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: memref<32xf32>,
// CHECK-SAME: %[[keysRaw:.*1]]: memref<2048xf32>,
// CHECK-SAME: %[[valuesRaw:.*2]]: memref<2048xf32>,
// CHECK-SAME: %[[lastValidKVIndexRaw:.*3]]: memref<1xi32>,
// CHECK-SAME: %[[outputRaw:.*4]]: memref<32xf32>)
// CHECK-SAME: attributes {mhal.arch = "[[$ARCH]]", rock.kernel}

// CHECK: rock.attention
// CHECK-NEXT: qk = %{{.*}} * %{{.*}}
// CHECK-NEXT: lastValidKVIndex = (%{{.*}} : memref<1xi32>)
// CHECK-NEXT: slidingWindowLookBack = 16
// CHECK: softmax(qk) * %{{.*}}
// CHECK: return

// L may reach seq_len_k - 1, the largest distance between two key positions.
// MAX-LOOK-BACK: rock.attention
// MAX-LOOK-BACK: lastValidKVIndex = (%{{.*}} : memref<1xi32>)
// MAX-LOOK-BACK: slidingWindowLookBack = 63

// A KV-cache bound without a look-back leaves the attribute off entirely.
// NON-SLIDING: rock.attention
// NON-SLIDING: lastValidKVIndex = (%{{.*}} : memref<1xi32>)
// NON-SLIDING-NOT: slidingWindowLookBack
// NON-SLIDING: softmax(qk)

// CHECK-LABEL: func.func @host_naive_attention
// Verify KV-cache masking is applied
// CHECK: tosa.matmul
// CHECK: tosa.greater
// CHECK: tosa.select

// Verify sliding window masking is applied in the CPU verifier:
// For inclusive index P and look-back L, lowerBound = max(0, P - L).
// Positions where col < lowerBound are then masked with -inf.
// CHECK: tosa.sub %{{.*}}, %{{.*}} : (tensor<1x1x1x64xi32>, tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xi32>
// CHECK: tosa.maximum %{{.*}}, %{{.*}} : (tensor<1x1x1x64xi32>, tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xi32>
// CHECK: tosa.greater %{{.*}}, %{{.*}} : (tensor<1x1x1x64xi32>, tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xi1>
// CHECK: tosa.select %{{.*}}, %{{.*}}, %{{.*}} : (tensor<1x1x1x64xi1>, tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>

// Verify softmax follows
// CHECK-DAG: tosa.reduce_max
// CHECK-DAG: tosa.exp
// CHECK-DAG: tosa.reduce_sum
// CHECK-DAG: tosa.reciprocal
// CHECK: tosa.matmul
// CHECK: return

// A fully masked row has max=-inf and exp-sum=0. Verify the CPU reference uses
// finite normalization operands while retaining the original values for LSE.
// SAFE-LABEL: func.func @host_naive_attention
// SAFE: %[[MAX:.*]] = tosa.reduce_max
// SAFE: %[[LOWEST:.*]] = "tosa.const"() <{values = dense<-3.40282347E+38> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// SAFE: %[[SAFE_MAX:.*]] = tosa.maximum %[[MAX]], %[[LOWEST]]
// SAFE: %[[NORMALIZED:.*]] = tosa.sub %{{.*}}, %[[SAFE_MAX]]
// SAFE: %[[EXPS:.*]] = tosa.exp %[[NORMALIZED]]
// SAFE: %[[SUM:.*]] = tosa.reduce_sum %[[EXPS]]
// SAFE: %[[MAX_FOR_LSE:.*]] = tosa.cast %[[MAX]]
// SAFE: %[[LOG_SUM:.*]] = tosa.log
// SAFE: tosa.add %[[LOG_SUM]], %[[MAX_FOR_LSE]]
// SAFE: %[[ONE:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// SAFE: %[[SAFE_SUM:.*]] = tosa.maximum %[[SUM]], %[[ONE]]
// SAFE: tosa.reciprocal %[[SAFE_SUM]]
// SAFE: tosa.matmul
// SAFE: return

// When last_valid_kv_index is omitted, use the last valid key position for every
// group so tuning-problem keys can be reconstructed by tuningRunner.
// DEFAULT-LAST-VALID-KV-INDEX-LABEL: func.func @rock_attention(
// DEFAULT-LAST-VALID-KV-INDEX-SAME: memref<2xi32>
// DEFAULT-LAST-VALID-KV-INDEX: lastValidKVIndex = (%{{.*}} : memref<2xi32>)
// DEFAULT-LAST-VALID-KV-INDEX: slidingWindowLookBack = 16
// DEFAULT-LAST-VALID-KV-INDEX-COUNT-2: arith.constant 63 : i32
