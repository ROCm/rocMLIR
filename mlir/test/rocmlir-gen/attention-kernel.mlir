// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_SCALE

// CHECK_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_SCALE-LABEL: func.func @rock_attention
// CHECK_SCALE-SAME: (%[[queriesRaw:.*0]]: memref<32768xf32>,
// CHECK_SCALE-SAME: %[[keysRaw:.*1]]: memref<32768xf32>,
// CHECK_SCALE-SAME: %[[valuesRaw:.*2]]: memref<32768xf32>,
// CHECK_SCALE-SAME: %[[scaleRaw:.*3]]: memref<1048576xf32>,
// CHECK_SCALE-SAME: %[[outputRaw:.*4]]: memref<32768xf32>)
// CHECK_SCALE-SAME: attributes {mhal.arch = "[[$ARCH]]", rock.kernel}
// CHECK_SCALE-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>
// CHECK_SCALE-NEXT: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<32768xf32> to memref<1x32x1024xf32>
// CHECK_SCALE-NEXT: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>
// CHECK_SCALE-NEXT: %[[scale:.*]] = rock.transform %[[scaleRaw]] {{.*}} : memref<1048576xf32> to memref<1x1024x1024xf32>
// CHECK_SCALE-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>

// CHECK_SCALE-NEXT: rock.attention
// CHECK_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_SCALE-NEXT: qk = elementwise otherIns(%[[scale]]
// CHECK_SCALE: %[[output]] = softmax(qk) * %[[values]]
// CHECK_SCALE: return

// CHECK_SCALE-LABEL: func.func @host_naive_attention
// CHECK_SCALE: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[sqkTensor:.*]] = tosa.mul %[[qkTensor]], %[[scaleTensor:.*]], %{{.*}} : ([[squareShape]], [[squareShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[sqkTensorCast:.*]] = tosa.cast %[[sqkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[sqkTensorCast]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[sqkTensorCast]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[valuesShape]]
// CHECK_SCALE: return

// ----

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_NO_SCALE

// CHECK_NO_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_NO_SCALE-LABEL: func.func @rock_attention
// CHECK_NO_SCALE-SAME: (%[[queriesRaw:.*0]]: memref<32768xf32>,
// CHECK_NO_SCALE-SAME: %[[keysRaw:.*1]]: memref<32768xf32>,
// CHECK_NO_SCALE-SAME: %[[valuesRaw:.*2]]: memref<32768xf32>,
// CHECK_NO_SCALE-SAME: %[[outputRaw:.*3]]: memref<32768xf32>)
// CHECK_NO_SCALE-SAME: attributes {mhal.arch = "[[$ARCH]]", rock.kernel}
// CHECK_NO_SCALE-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<32768xf32> to memref<1x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>

// CHECK_NO_SCALE-NEXT: rock.attention
// CHECK_NO_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_NO_SCALE: %[[output]] = softmax(qk) * %[[values]]
// CHECK_NO_SCALE: return

// CHECK_NO_SCALE-LABEL: func.func @host_naive_attention
// CHECK_NO_SCALE: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]
// CHECK_NO_SCALE: %[[qkTensorCast:.*]] = tosa.cast %[[qkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[qkTensorCast]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_NO_SCALE-DAG: %[[normilizedQkTensor:.*]] = tosa.sub %[[qkTensorCast]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedQkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_NO_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_NO_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[valuesShape]]
// CHECK_NO_SCALE: return

// ----

// Per-tensor transpose flags. The baseline (no flag) layout is:
//   Q in [seq_q, head_qk], K in [head_qk, seq_k],
//   V in [seq_k, head_v],  O in [seq_q, head_v].
// `-transQ` / `-transV` flip the trailing two dims of the corresponding
// operand and surface a `tr` modifier on the matmul; `-transO` flips the
// output, which surfaces a `tr` modifier on the second-gemm result inside
// `rock.attention`.

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 -transQ | FileCheck %s --enable-var-scope --check-prefix=TRANS_Q
// TRANS_Q-LABEL: func.func @rock_attention
// Q (flat 4096) becomes head_qk x seq_q instead of seq_q x head_qk.
// TRANS_Q: rock.transform %{{.*}} : memref<4096xf16> to memref<1x32x128xf16>
// TRANS_Q: rock.transform %{{.*}} : memref<8192xf16> to memref<1x32x256xf16>
// TRANS_Q: rock.transform %{{.*}} : memref<16384xf16> to memref<1x256x64xf16>
// TRANS_Q: rock.attention
// TRANS_Q: qk = tr %{{.*}} * %{{.*}} : memref<1x32x128xf16>, memref<1x32x256xf16>

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 -transV | FileCheck %s --enable-var-scope --check-prefix=TRANS_V
// TRANS_V-LABEL: func.func @rock_attention
// V (flat 16384) becomes head_v x seq_k instead of seq_k x head_v.
// TRANS_V: rock.transform %{{.*}} : memref<4096xf16> to memref<1x128x32xf16>
// TRANS_V: rock.transform %{{.*}} : memref<8192xf16> to memref<1x32x256xf16>
// TRANS_V: rock.transform %{{.*}} : memref<16384xf16> to memref<1x64x256xf16>
// TRANS_V: rock.attention
// TRANS_V: softmax(qk) * tr %{{.*}} : memref<1x64x256xf16>

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 -transO | FileCheck %s --enable-var-scope --check-prefix=TRANS_O
// TRANS_O-LABEL: func.func @rock_attention
// TRANS_O: rock.attention
// O (flat 8192) becomes head_v x seq_q instead of seq_q x head_v.
// TRANS_O: tr %{{.*}} = softmax(qk) * %{{.*}} : memref<1x256x64xf16> -> memref<1x64x128xf16>

// ----

// Attention bias fusion. The bias is a pre-softmax elementwise add whose input
// has the [seq_q, seq_k] score layout, so it feeds `rock.attention` as an
// `otherIns` of the pre-softmax `elementwise` region.

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 --with-attn-bias | FileCheck %s --enable-var-scope --check-prefix=BIAS
// BIAS-LABEL: func.func @rock_attention
// Without -transBias the bias argument is already laid out as seq_q x seq_k and
// is fed to the pre-softmax elementwise add directly (no extra transpose).
// BIAS: %[[bias:.*]] = rock.transform %{{.*}} : memref<32768xf16> to memref<1x128x256xf16>
// BIAS: rock.attention
// BIAS: qk = elementwise otherIns(%[[bias]] : memref<1x128x256xf16>)

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 --with-attn-bias --transBias | FileCheck %s --enable-var-scope --check-prefix=TRANS_BIAS
// TRANS_BIAS-LABEL: func.func @rock_attention
// With -transBias the bias argument is materialized transposed in memory
// (seq_k x seq_q), then a rank-preserving permutation restores the logical
// seq_q x seq_k layout before it feeds the pre-softmax elementwise add.
// TRANS_BIAS: %[[biasRaw:.*]] = rock.transform %{{.*}} : memref<32768xf16> to memref<1x256x128xf16>
// TRANS_BIAS: %[[bias:.*]] = rock.transform %[[biasRaw]] {{.*}} : memref<1x256x128xf16> to memref<1x128x256xf16>
// TRANS_BIAS: rock.attention
// TRANS_BIAS: qk = elementwise otherIns(%[[bias]] : memref<1x128x256xf16>)

// The host reference must mirror the transpose so validation stays correct: the
// seq_k x seq_q bias is permuted to seq_q x seq_k before the pre-softmax add.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 128 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 64 -t f16 --with-attn-bias --transBias -pv | FileCheck %s --enable-var-scope --check-prefix=TRANS_BIAS_HOST
// TRANS_BIAS_HOST-LABEL: func.func @host_naive_attention
// TRANS_BIAS_HOST: memref.expand_shape %{{.*}} : memref<32768xf16> into memref<1x256x128xf16>
// TRANS_BIAS_HOST: linalg.transpose ins(%{{.*}} : memref<1x256x128xf16>) outs(%{{.*}} : memref<1x128x256xf16>) permutation = [0, 2, 1]
