// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -current_seq_len=33 -sliding_window_size=16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: memref<32xf32>,
// CHECK-SAME: %[[keysRaw:.*1]]: memref<2048xf32>,
// CHECK-SAME: %[[valuesRaw:.*2]]: memref<2048xf32>,
// CHECK-SAME: %[[currentSeqLenRaw:.*3]]: memref<1xi32>,
// CHECK-SAME: %[[outputRaw:.*4]]: memref<32xf32>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// CHECK: rock.attention
// CHECK-NEXT: qk = %{{.*}} * %{{.*}}
// CHECK-NEXT: currentSeqLen = (%{{.*}} : memref<1xi32>)
// CHECK-NEXT: slidingWindowSize = 16
// CHECK: softmax(qk) * %{{.*}}
// CHECK: return

// CHECK-LABEL: func.func @host_naive_attention
// Verify KV-cache masking is applied
// CHECK: tosa.matmul
// CHECK: tosa.greater
// CHECK: tosa.select

// Verify sliding window masking is applied in the CPU verifier:
// The sliding window masking computes lowerBound = max(0, currentSeqLen - windowSize),
// then masks positions where col < lowerBound with -inf.
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
