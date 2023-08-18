// RUN: rocmlir-gen --arch %arch --operation attention -seq_len 1024 -num_heads 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queries:.*0]]: memref<1024x32xf32>,
// CHECK-SAME: %[[keys:.*1]]: memref<32x1024xf32>,
// CHECK-SAME: %[[values:.*2]]: memref<1024x32xf32>,
// CHECK-SAME: %[[scale:.*3]]: memref<1024x1024xf32>,
// CHECK-SAME: %[[output:.*4]]: memref<1024x32xf32>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// CHECK-NEXT: rock.attention(%[[queries]], %[[keys]], %[[values]], %[[scale]], %[[output]])
// CHECK: return

// CHECK-LABEL: func.func @host_naive_attention
// CHECK: %[[qkTensor:.*]] = "tosa.matmul"(%[[queriesTensor:.*]], %[[keysTensor:.*]]) : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]]) -> [[squareShape:tensor<.*>]]
// CHECK-NEXT: %[[sqkTensor:.*]] = "tosa.mul"(%[[qkTensor]], %[[scaleTensor:.*]]) <{{.*}}> : ([[squareShape]], [[squareShape]]) -> [[squareShape]]
// CHECK-NEXT: %[[sqkMaxs:.*]] = "tosa.reduce_max"(%[[sqkTensor]]) <{{.*}}> : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK-NEXT: %[[normilizedSqkTensor:.*]] = "tosa.sub"(%[[sqkTensor]], %[[sqkMaxs]]) : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK-NEXT: %[[expsTensor:.*]] = "tosa.exp"(%[[normilizedSqkTensor]]) : ([[squareShape]]) -> [[squareShape]]
// CHECK-NEXT: %[[expsSumsTensor:.*]] = "tosa.reduce_sum"(%[[expsTensor]]) <{{.*}}> : ([[squareShape]]) -> [[reducedShape]]
// CHECK-NEXT: %[[invExpsSums:.*]] = "tosa.reciprocal"(%[[expsSumsTensor]]) : ([[reducedShape]]) -> [[reducedShape]]
// CHECK-NEXT: %[[softmaxTensor:.*]] = "tosa.mul"(%[[expsTensor]], %[[invExpsSums]]) <{{.*}}> : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK-NEXT: %[[resultTensor:.*]] = "tosa.matmul"(%[[softmaxTensor]], %[[valuesTensor:.*]]) : ([[squareShape]], [[valuesShape:tensor<.*>]]) -> [[valuesShape]]
// CHECK: return
