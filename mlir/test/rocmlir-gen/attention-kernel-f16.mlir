// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f16 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: memref<32768xf16>,
// CHECK-SAME: %[[keysRaw:.*1]]: memref<32768xf16>,
// CHECK-SAME: %[[valuesRaw:.*2]]: memref<32768xf16>,
// CHECK-SAME: %[[scaleRaw:.*3]]: memref<1048576xf16>,
// CHECK-SAME: %[[outputRaw:.*4]]: memref<32768xf16>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<32768xf16> to memref<1x1024x32xf16>
// CHECK-NEXT: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<32768xf16> to memref<1x32x1024xf16>
// CHECK-NEXT: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<32768xf16> to memref<1x1024x32xf16>
// CHECK-NEXT: %[[scale:.*]] = rock.transform %[[scaleRaw]] {{.*}} : memref<1048576xf16> to memref<1x1024x1024xf16>
// CHECK-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<32768xf16> to memref<1x1024x32xf16>

// CHECK-NEXT: rock.attention
// CHECK-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK-NEXT: qk = elementwise otherIns(%[[scale]]
// CHECK: %[[output]] = softmax(qk) * %[[values]]
// CHECK: return

// CHECK-LABEL: func.func @host_naive_attention
// CHECK: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf16>, tensor<1xf16>) -> [[squareShape:tensor<.*>]]
// CHECK-DAG: %[[sqkTensor:.*]] = tosa.mul %[[qkTensor]], %[[scaleTensor:.*]], %{{.*}} : ([[squareShape]], [[squareShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK-DAG: %[[sqkTensorCast:.*]] = tosa.cast %[[sqkTensor]] : ([[squareShape]]) -> [[squareShapeF32:tensor<.*>]]
// CHECK-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[sqkTensorCast]] {{.*}} : ([[squareShapeF32]]) -> [[reducedShape:tensor<.*>]]
// CHECK-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[sqkTensorCast]], %[[sqkMaxs]] : ([[squareShapeF32]], [[reducedShape]]) -> [[squareShapeF32]]
// CHECK-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : ([[squareShapeF32]]) -> [[squareShapeF32]]
// CHECK-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShapeF32]]) -> [[reducedShape]]
// CHECK-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShapeF32]], [[reducedShape]], tensor<1xi8>) -> [[squareShapeF32]] 
// CHECK-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShapeF32]]) -> [[squareShape]]
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf16>, tensor<1xf16>) -> [[squareShape:tensor<.*>]]
// CHECK: return
