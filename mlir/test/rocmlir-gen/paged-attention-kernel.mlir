// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f16 -pv --apply-bufferization-pipeline=false --paged-attention --num-pages 32 --page-size 1024 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// Verify paged attention kernel signature:
// - Q input: memref<32768xf16> (flattened [1, 1024, 32])
// - K page table: memref<32xi64> (page pointers, numPages = 32)
// - V page table: memref<32xi64> (page pointers, numPages = 32)
// - Output: memref<32768xf16> (flattened [1, 1024, 32])
// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: memref<32768xf16>,
// CHECK-SAME: %[[keysPageTable:.*1]]: memref<32xi64>,
// CHECK-SAME: %[[valuesPageTable:.*2]]: memref<32xi64>,
// CHECK-SAME: %[[outputRaw:.*3]]: memref<32768xf16>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// Transform Q to [G, seq_q, head_qk]
// CHECK-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<32768xf16> to memref<1x1024x32xf16>

// Transform K page table to [batch, numPages, 1]
// CHECK-NEXT: %[[keysPageTableTransformed:.*]] = rock.transform %[[keysPageTable]] {{.*}} : memref<32xi64> to memref<1x32x1xi64>

// Transform V page table to [batch, numPages, 1]
// CHECK-NEXT: %[[valuesPageTableTransformed:.*]] = rock.transform %[[valuesPageTable]] {{.*}} : memref<32xi64> to memref<1x32x1xi64>

// Transform output to [G, seq_q, head_v]
// CHECK-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<32768xf16> to memref<1x1024x32xf16>

// rock.deref: dereference K page table to get actual K data
// CHECK-NEXT: %[[keyDeref:.*]] = rock.deref %[[keysPageTableTransformed]] : memref<1x32x1xi64> -> memref<1x32x1024xf16>

// rock.deref: dereference V page table to get actual V data
// CHECK-NEXT: %[[valueDeref:.*]] = rock.deref %[[valuesPageTableTransformed]] : memref<1x32x1xi64> -> memref<1x32x1024xf16>

// Transform deref'd K to intermediate shapes for attention GEMM
// CHECK-NEXT: %[[keyTransform1:.*]] = rock.transform %[[keyDeref]] {{.*}} : memref<1x32x1024xf16> to memref<1x32768xf16>
// CHECK-NEXT: %[[keyTransform2:.*]] = rock.transform %[[keyTransform1]] {{.*}} : memref<1x32768xf16> to memref<1x1x1024x32xf16>
// CHECK-NEXT: %[[keys:.*]] = rock.transform %[[keyTransform2]] {{.*}} : memref<1x1x1024x32xf16> to memref<1x32x1024xf16>

// Transform deref'd V to intermediate shapes for attention GEMM
// CHECK-NEXT: %[[valueTransform1:.*]] = rock.transform %[[valueDeref]] {{.*}} : memref<1x32x1024xf16> to memref<1x32768xf16>
// CHECK-NEXT: %[[valueTransform2:.*]] = rock.transform %[[valueTransform1]] {{.*}} : memref<1x32768xf16> to memref<1x1x1024x32xf16>
// CHECK-NEXT: %[[values:.*]] = rock.transform %[[valueTransform2]] {{.*}} : memref<1x1x1024x32xf16> to memref<1x1024x32xf16>

// Verify rock.attention op with keyAddresses and valueAddresses attributes
// CHECK-NEXT: rock.attention
// CHECK-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK-NEXT: keyAddresses = (%[[keyDeref]] : memref<1x32x1024xf16>)
// CHECK-NEXT: valueAddresses = (%[[valueDeref]] : memref<1x32x1024xf16>)
// CHECK: %[[output]] = softmax(qk) * %[[values]]
// CHECK: return

// =============================================================================
// CPU host function validation for paged attention
// =============================================================================

// CHECK-LABEL: func.func @host_naive_attention
// CHECK-SAME: (%[[hostQ:.*0]]: memref<32768xf16>,
// CHECK-SAME: %[[hostK:.*1]]: memref<32768xf16>,
// CHECK-SAME: %[[hostV:.*2]]: memref<32768xf16>,
// CHECK-SAME: %[[hostOut:.*3]]: memref<32768xf16>)

// Convert Q memref to tensor and reshape to [1, 1024, 32]
// CHECK: bufferization.to_tensor %[[hostQ]]
// CHECK: tosa.reshape {{.*}} : (tensor<32768xf16>, !tosa.shape<3>) -> tensor<1x1024x32xf16>

// Reshape K to [1, 32, 1024] for Q*K^T matmul
// CHECK: bufferization.to_tensor %[[hostK]]
// CHECK: tosa.reshape {{.*}} : (tensor<32768xf16>, !tosa.shape<3>) -> tensor<1x32x1024xf16>

// Reshape V to [1, 1024, 32]
// CHECK: bufferization.to_tensor %[[hostV]]
// CHECK: tosa.reshape {{.*}} : (tensor<32768xf16>, !tosa.shape<3>) -> tensor<1x1024x32xf16>

// First matmul: Q * K^T -> [1, 1024, 1024]
// CHECK: tosa.matmul {{.*}} {acc_type = f32} : (tensor<1x1024x32xf16>, tensor<1x32x1024xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x1024x1024xf16>

// Softmax: cast to f32, reduce_max, sub, exp, reduce_sum, reciprocal, mul, cast back
// CHECK: tosa.cast {{.*}} : (tensor<1x1024x1024xf16>) -> tensor<1x1024x1024xf32>
// CHECK: tosa.reduce_max {{.*}} : (tensor<1x1024x1024xf32>) -> tensor<1x1024x1xf32>
// CHECK: tosa.sub {{.*}} : (tensor<1x1024x1024xf32>, tensor<1x1024x1xf32>) -> tensor<1x1024x1024xf32>
// CHECK: tosa.exp {{.*}} : (tensor<1x1024x1024xf32>) -> tensor<1x1024x1024xf32>
// CHECK: tosa.reduce_sum {{.*}} : (tensor<1x1024x1024xf32>) -> tensor<1x1024x1xf32>
// CHECK: tosa.reciprocal {{.*}} : (tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
// CHECK: tosa.mul {{.*}} : (tensor<1x1024x1024xf32>, tensor<1x1024x1xf32>, tensor<1xi8>) -> tensor<1x1024x1024xf32>
// CHECK: tosa.cast {{.*}} : (tensor<1x1024x1024xf32>) -> tensor<1x1024x1024xf16>

// Second matmul: softmax(Q*K^T) * V -> [1, 1024, 32]
// CHECK: tosa.matmul {{.*}} {acc_type = f32} : (tensor<1x1024x1024xf16>, tensor<1x1024x32xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x1024x32xf16>

// Reshape output and copy to result
// CHECK: tosa.reshape {{.*}} : (tensor<1x1024x32xf16>, !tosa.shape<1>) -> tensor<32768xf16>
// CHECK: bufferization.to_buffer
// CHECK: memref.copy
// CHECK: return

// ----

// Test paged attention with GQA (grouped query attention)
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f16 -pv --apply-bufferization-pipeline=false --paged-attention --num-pages 64 --page-size 1024 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_GQA

// CHECK_GQA: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// Verify GQA paged attention kernel signature:
// - Q input: memref<131072xf16> (flattened [4, 1024, 32])
// - K page table: memref<64xi64> (page pointers for 2 heads)
// - V page table: memref<64xi64> (page pointers for 2 heads)
// - Output: memref<131072xf16> (flattened [4, 1024, 32])
// CHECK_GQA-LABEL: func.func @rock_attention
// CHECK_GQA-SAME: (%[[queriesRaw:.*0]]: memref<131072xf16>,
// CHECK_GQA-SAME: %[[keysPageTable:.*1]]: memref<64xi64>,
// CHECK_GQA-SAME: %[[valuesPageTable:.*2]]: memref<64xi64>,
// CHECK_GQA-SAME: %[[outputRaw:.*3]]: memref<131072xf16>)
// CHECK_GQA-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// Transform Q to [G, seq_q, head_qk] with G = num_heads_q = 4
// CHECK_GQA-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<131072xf16> to memref<4x1024x32xf16>

// Transform K page table
// CHECK_GQA-NEXT: %[[keysPageTableTransformed:.*]] = rock.transform %[[keysPageTable]] {{.*}} : memref<64xi64> to memref<1x64x1xi64>

// Transform V page table
// CHECK_GQA-NEXT: %[[valuesPageTableTransformed:.*]] = rock.transform %[[valuesPageTable]] {{.*}} : memref<64xi64> to memref<1x64x1xi64>

// Transform output
// CHECK_GQA-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<131072xf16> to memref<4x1024x32xf16>

// rock.deref K
// CHECK_GQA-NEXT: %[[keyDeref:.*]] = rock.deref %[[keysPageTableTransformed]] : memref<1x64x1xi64> -> memref<1x64x1024xf16>

// rock.deref V
// CHECK_GQA-NEXT: %[[valueDeref:.*]] = rock.deref %[[valuesPageTableTransformed]] : memref<1x64x1xi64> -> memref<1x64x1024xf16>

// K transforms to [G, head_dim_qk, seq_k] with G = num_heads_kv = 2
// CHECK_GQA: %[[keys:.*]] = rock.transform %{{.*}} {{.*}} to memref<2x32x1024xf16>

// V transforms to [G, seq_k, head_dim_v] with G = num_heads_kv = 2
// CHECK_GQA: %[[values:.*]] = rock.transform %{{.*}} {{.*}} to memref<2x1024x32xf16>

// Verify rock.attention op with GQA and paged attention
// CHECK_GQA: rock.attention
// CHECK_GQA-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_GQA-NEXT: keyAddresses = (%[[keyDeref]] : memref<1x64x1024xf16>)
// CHECK_GQA-NEXT: valueAddresses = (%[[valueDeref]] : memref<1x64x1024xf16>)
// CHECK_GQA: %[[output]] = softmax(qk) * %[[values]]
// CHECK_GQA-NEXT: numHeadsKV = 2 : i32, numHeadsQ = 4 : i32
// CHECK_GQA: return
