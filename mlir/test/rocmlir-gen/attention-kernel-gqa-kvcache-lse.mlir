// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -current_seq_len=33 -return_lse -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: memref<131072xf32>,
// CHECK-SAME: %[[keysRaw:.*1]]: memref<65536xf32>,
// CHECK-SAME: %[[valuesRaw:.*2]]: memref<65536xf32>,
// CHECK-SAME: %[[currentSeqLenRaw:.*3]]: memref<1xi32>,
// CHECK-SAME: %[[lseRaw:.*4]]: memref<4096xf32>,
// CHECK-SAME: %[[outputRaw:.*5]]: memref<131072xf32>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK-NEXT: %[[keysGQA:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<65536xf32> to memref<2x32x1024xf32>
// CHECK-NEXT: %[[valuesGQA:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<65536xf32> to memref<2x1024x32xf32>
// CHECK-NEXT: %[[currentSeqLen:.*]] = rock.transform %[[currentSeqLenRaw]] {{.*}} : memref<1xi32> to memref<1xi32>
// CHECK-NEXT: %[[lse:.*]] = rock.transform %[[lseRaw]] {{.*}} : memref<4096xf32> to memref<4x1024xf32>
// CHECK-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK-NEXT: %[[currentSeqLenAddDim:.*]] = rock.transform %[[currentSeqLen]] {{.*}} : memref<1xi32> to memref<1x1xi32>
// CHECK-NEXT: %[[currentSeqLenBroadcast:.*]] = rock.transform %[[currentSeqLenAddDim]] {{.*}} : memref<1x1xi32> to memref<1x4xi32>
// CHECK-NEXT: %[[currentSeqLenMerge:.*]] = rock.transform %[[currentSeqLenBroadcast]] {{.*}} : memref<1x4xi32> to memref<4xi32>
// CHECK-NEXT: %[[keysAddDim:.*]] = rock.transform %[[keysGQA]] {{.*}} : memref<2x32x1024xf32> to memref<2x1x32x1024xf32>
// CHECK-NEXT: %[[keysBroadcast:.*]] = rock.transform %[[keysAddDim]] {{.*}} : memref<2x1x32x1024xf32> to memref<2x2x32x1024xf32>
// CHECK-NEXT: %[[keys:.*]] = rock.transform %[[keysBroadcast]] {{.*}} : memref<2x2x32x1024xf32> to memref<4x32x1024xf32>
// CHECK-NEXT: %[[valuesAddDim:.*]] = rock.transform %[[valuesGQA]] {{.*}} : memref<2x1024x32xf32> to memref<2x1x1024x32xf32>
// CHECK-NEXT: %[[valuesBroadcast:.*]] = rock.transform %[[valuesAddDim]] {{.*}} : memref<2x1x1024x32xf32> to memref<2x2x1024x32xf32>
// CHECK-NEXT: %[[values:.*]] = rock.transform %[[valuesBroadcast]] {{.*}} : memref<2x2x1024x32xf32> to memref<4x1024x32xf32>

// CHECK-NEXT: rock.attention
// CHECK-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK-NEXT: currentSeqLen = (%[[currentSeqLenMerge]] : memref<4xi32>)
// CHECK-NEXT: lse = %[[lse]] : memref<4x1024xf32>
// CHECK: %[[output]] = softmax(qk) * %[[values]]
// CHECK: return

// CHECK-LABEL: func.func @host_naive_attention
// CHECK: %[[keysExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 32, 1024] : tensor<2x32x1024xf32> into tensor<2x1x32x1024xf32>
// CHECK: %[[keysAdd:.*]] = tosa.add %{{.*}}, %[[keysExpanded]] : (tensor<2x2x32x1024xf32>, tensor<2x1x32x1024xf32>) -> tensor<2x2x32x1024xf32>
// CHECK: %[[keysTensor:.*]] = tensor.collapse_shape %[[keysAdd]] {{.*}} : tensor<2x2x32x1024xf32> into tensor<4x32x1024xf32>
// CHECK: %[[valuesExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 1024, 32] : tensor<2x1024x32xf32> into tensor<2x1x1024x32xf32>
// CHECK: %[[valuesAdd:.*]] = tosa.add %{{.*}}, %[[valuesExpanded]] : (tensor<2x2x1024x32xf32>, tensor<2x1x1024x32xf32>) -> tensor<2x2x1024x32xf32>
// CHECK: %[[valuesTensor:.*]] = tensor.collapse_shape %[[valuesAdd]] {{.*}} : tensor<2x2x1024x32xf32> into [[valuesShape:tensor<.*>]]
// CHECK: %[[qkTensorOrig:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]

// CHECK-DAG: %[[qkTensorCast:.*]] = tosa.cast %[[qkTensorOrig]] : (tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
// CHECK-DAG: %[[currSeqLenTensorDumbReshaped:.*]] = tosa.reshape %[[currSeqLenTensor:.*]], %{{.*}} : (tensor<1xi32>, !tosa.shape<1>) -> tensor<1xi32>
// CHECK-DAG: %[[currSeqLenTensorReshaped:.*]] = tosa.reshape %[[currSeqLenTensorDumbReshaped]], %{{.*}} : (tensor<1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
// CHECK-DAG: %[[qkTensorReshaped:.*]] = tosa.reshape %[[qkTensorCast]], %{{.*}} : (tensor<4x1024x1024xf32>, !tosa.shape<4>) -> tensor<1x4x1024x1024xf32>
// CHECK-DAG: %[[range:.*]] = "tosa.const"() <{values = {{.*}} : tensor<1024xi32>}> : () -> tensor<1024xi32>
// CHECK-DAG: %[[rangeReshaped:.*]] = tosa.reshape %[[range]], %{{.*}} : (tensor<1024xi32>, !tosa.shape<4>) -> tensor<1x1x1x1024xi32>
// CHECK-DAG: %[[zero:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x1024x1024xi32>}> : () -> tensor<1x4x1024x1024xi32>
// CHECK-DAG: %[[rangeBroadcast:.*]] = tosa.add %[[zero]], %[[rangeReshaped]] : (tensor<1x4x1024x1024xi32>, tensor<1x1x1x1024xi32>) -> tensor<1x4x1024x1024xi32>
// CHECK-DAG: %[[zero2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x1024x1024xi32>}> : () -> tensor<1x4x1024x1024xi32>
// CHECK-DAG: %[[currSeqLenTensorBroadcast:.*]] = tosa.add %[[zero2]], %[[currSeqLenTensorReshaped]] : (tensor<1x4x1024x1024xi32>, tensor<1x1x1x1xi32>) -> tensor<1x4x1024x1024xi32>
// CHECK-DAG: %[[mask:.*]] = tosa.greater %[[rangeBroadcast]], %[[currSeqLenTensorBroadcast]] : (tensor<1x4x1024x1024xi32>, tensor<1x4x1024x1024xi32>) -> tensor<1x4x1024x1024xi1>
// CHECK-DAG: %[[negInf:.*]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x4x1024x1024xf32>}> : () -> tensor<1x4x1024x1024xf32>
// CHECK-DAG: %[[qkTensorBeforeReshape:.*]] = tosa.select %[[mask]], %[[negInf]], %[[qkTensorReshaped]] : (tensor<1x4x1024x1024xi1>, tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
// CHECK-DAG: %[[qkTensor:.*]] = tosa.reshape %[[qkTensorBeforeReshape]], %{{.*}} : (tensor<1x4x1024x1024xf32>, !tosa.shape<3>) -> tensor<4x1024x1024xf32>

// CHECK-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[qkTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK-DAG: %[[normilizedQkTensor:.*]] = tosa.sub %[[qkTensor]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedQkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]

// CHECK-DAG: %[[expsSumsTensorCast:.*]] = tosa.cast %[[expsSumsTensor]] : (tensor<4x1024x1xf32>) -> tensor<4x1024x1xf32>
// CHECK-DAG: %[[sqkMaxsCast:.*]] = tosa.cast %[[sqkMaxs]] : (tensor<4x1024x1xf32>) -> tensor<4x1024x1xf32>
// CHECK-DAG: %[[logL:.*]] = tosa.log %[[expsSumsTensorCast]] : (tensor<4x1024x1xf32>) -> tensor<4x1024x1xf32>
// CHECK-DAG: %[[resultLse:.*]] = tosa.add %[[logL]], %[[sqkMaxsCast]] : (tensor<4x1024x1xf32>, tensor<4x1024x1xf32>) -> tensor<4x1024x1xf32>

// CHECK-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> tensor<4x1024x32xf32>
// CHECK: return
