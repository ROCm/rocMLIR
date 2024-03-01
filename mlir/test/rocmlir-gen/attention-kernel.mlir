// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_SCALE
// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_NO_SCALE

// CHECK_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_SCALE-LABEL: func.func @rock_attention
// CHECK_SCALE-SAME: (%[[queries:.*0]]: memref<1x1024x32xf32>,
// CHECK_SCALE-SAME: %[[keys:.*1]]: memref<1x32x1024xf32>,
// CHECK_SCALE-SAME: %[[values:.*2]]: memref<1x1024x32xf32>,
// CHECK_SCALE-SAME: %[[scale:.*3]]: memref<1x1024x1024xf32>,
// CHECK_SCALE-SAME: %[[output:.*4]]: memref<1x1024x32xf32>)
// CHECK_SCALE-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// CHECK_SCALE-NEXT: rock.attention
// CHECK_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_SCALE-NEXT: qk = elementwise otherIns(%[[scale]]
// CHECK_SCALE: %[[output]] = softmax(qk) * %[[values]]
// CHECK_SCALE: return

// CHECK_SCALE-LABEL: func.func @host_naive_attention
// CHECK_SCALE: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]] : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]]) -> [[squareShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[sqkTensor:.*]] = tosa.mul %[[qkTensor]], %[[scaleTensor:.*]] {{.*}} : ([[squareShape]], [[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[sqkTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[sqkTensor]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]] {{.*}} : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensor]], %[[valuesTensor:.*]] : ([[squareShape]], [[valuesShape:tensor<.*>]]) -> [[valuesShape]]
// CHECK_SCALE: return

// ----

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_NO_SCALE

// CHECK_NO_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_NO_SCALE-LABEL: func.func @rock_attention
// CHECK_NO_SCALE-SAME: (%[[queries:.*0]]: memref<1x1024x32xf32>,
// CHECK_NO_SCALE-SAME: %[[keys:.*1]]: memref<1x32x1024xf32>,
// CHECK_NO_SCALE-SAME: %[[values:.*2]]: memref<1x1024x32xf32>,
// CHECK_NO_SCALE-SAME: %[[output:.*3]]: memref<1x1024x32xf32>)
// CHECK_NO_SCALE-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// CHECK_NO_SCALE-NEXT: rock.attention
// CHECK_NO_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_NO_SCALE: %[[output]] = softmax(qk) * %[[values]]
// CHECK_NO_SCALE: return

// CHECK_NO_SCALE-LABEL: func.func @host_naive_attention
// CHECK_NO_SCALE: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]] : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]]) -> [[squareShape:tensor<.*>]]
// CHECK_NO_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[qkTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_NO_SCALE-DAG: %[[normilizedQkTensor:.*]] = tosa.sub %[[qkTensor]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedQkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_NO_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_NO_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]] {{.*}} : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensor]], %[[valuesTensor:.*]] : ([[squareShape]], [[valuesShape:tensor<.*>]]) -> [[valuesShape]]
// CHECK_NO_SCALE: return
