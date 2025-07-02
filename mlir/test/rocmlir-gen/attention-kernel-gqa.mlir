// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_SCALE

// CHECK_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_SCALE-LABEL: func.func @rock_attention
// CHECK_SCALE-SAME: (%[[queriesRaw:.*0]]: memref<131072xf32>,
// CHECK_SCALE-SAME: %[[keysRaw:.*1]]: memref<65536xf32>,
// CHECK_SCALE-SAME: %[[valuesRaw:.*2]]: memref<65536xf32>,
// CHECK_SCALE-SAME: %[[scaleRaw:.*3]]: memref<4194304xf32>,
// CHECK_SCALE-SAME: %[[outputRaw:.*4]]: memref<131072xf32>)
// CHECK_SCALE-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK_SCALE-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK_SCALE-NEXT: %[[keysGQA:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<65536xf32> to memref<2x32x1024xf32>
// CHECK_SCALE-NEXT: %[[valuesGQA:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<65536xf32> to memref<2x1024x32xf32>
// CHECK_SCALE-NEXT: %[[scale:.*]] = rock.transform %[[scaleRaw]] {{.*}} : memref<4194304xf32> to memref<4x1024x1024xf32>
// CHECK_SCALE-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK_SCALE-NEXT: %[[keysAddDim:.*]] = rock.transform %[[keysGQA]] {{.*}} : memref<2x32x1024xf32> to memref<2x1x32x1024xf32>
// CHECK_SCALE-NEXT: %[[keysBroadcast:.*]] = rock.transform %[[keysAddDim]] {{.*}} : memref<2x1x32x1024xf32> to memref<2x2x32x1024xf32>
// CHECK_SCALE-NEXT: %[[keys:.*]] = rock.transform %[[keysBroadcast]] {{.*}} : memref<2x2x32x1024xf32> to memref<4x32x1024xf32>
// CHECK_SCALE-NEXT: %[[valuesAddDim:.*]] = rock.transform %[[valuesGQA]] {{.*}} : memref<2x1024x32xf32> to memref<2x1x1024x32xf32>
// CHECK_SCALE-NEXT: %[[valuesBroadcast:.*]] = rock.transform %[[valuesAddDim]] {{.*}} : memref<2x1x1024x32xf32> to memref<2x2x1024x32xf32>
// CHECK_SCALE-NEXT: %[[values:.*]] = rock.transform %[[valuesBroadcast]] {{.*}} : memref<2x2x1024x32xf32> to memref<4x1024x32xf32>

// CHECK_SCALE-NEXT: rock.attention
// CHECK_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_SCALE-NEXT: qk = elementwise otherIns(%[[scale]]
// CHECK_SCALE: %[[output]] = softmax(qk) * %[[values]]
// CHECK_SCALE: return

// CHECK_SCALE-LABEL: func.func @host_naive_attention
// CHECK_SCALE: %[[keysExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 32, 1024] : tensor<2x32x1024xf32> into tensor<2x1x32x1024xf32>
// CHECK_SCALE: %[[keysAdd:.*]] = tosa.add %{{.*}}, %[[keysExpanded]] : (tensor<2x2x32x1024xf32>, tensor<2x1x32x1024xf32>) -> tensor<2x2x32x1024xf32>
// CHECK_SCALE: %[[keysTensor:.*]] = tensor.collapse_shape %[[keysAdd]] {{.*}} : tensor<2x2x32x1024xf32> into tensor<4x32x1024xf32>
// CHECK_SCALE: %[[valuesExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 1024, 32] : tensor<2x1024x32xf32> into tensor<2x1x1024x32xf32>
// CHECK_SCALE: %[[valuesAdd:.*]] = tosa.add %{{.*}}, %[[valuesExpanded]] : (tensor<2x2x1024x32xf32>, tensor<2x1x1024x32xf32>) -> tensor<2x2x1024x32xf32>
// CHECK_SCALE: %[[valuesTensor:.*]] = tensor.collapse_shape %[[valuesAdd]] {{.*}} : tensor<2x2x1024x32xf32> into tensor<4x1024x32xf32>
// CHECK_SCALE: %[[qkTensor:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor]], %{{.*}}, %{{.*}} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[sqkTensor:.*]] = tosa.mul %[[qkTensor]], %[[scaleTensor:.*]], %{{.*}} : ([[squareShape]], [[squareShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[sqkTensorCast:.*]] = tosa.cast %[[sqkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[sqkTensorCast]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[sqkTensorCast]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor]], %{{.*}}, %{{.*}} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[valuesShape]]
// CHECK_SCALE: return

// ----

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_NO_SCALE

// CHECK_NO_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_NO_SCALE-LABEL: func.func @rock_attention
// CHECK_NO_SCALE-SAME: (%[[queriesRaw:.*0]]: memref<131072xf32>,
// CHECK_NO_SCALE-SAME: %[[keysRaw:.*1]]: memref<65536xf32>,
// CHECK_NO_SCALE-SAME: %[[valuesRaw:.*2]]: memref<65536xf32>,
// CHECK_NO_SCALE-SAME: %[[outputRaw:.*3]]: memref<131072xf32>)
// CHECK_NO_SCALE-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK_NO_SCALE-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[keysGQA:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<65536xf32> to memref<2x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[valuesGQA:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<65536xf32> to memref<2x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[keysAddDim:.*]] = rock.transform %[[keysGQA]] {{.*}} : memref<2x32x1024xf32> to memref<2x1x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[keysBroadcast:.*]] = rock.transform %[[keysAddDim]] {{.*}} : memref<2x1x32x1024xf32> to memref<2x2x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[keys:.*]] = rock.transform %[[keysBroadcast]] {{.*}} : memref<2x2x32x1024xf32> to memref<4x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[valuesAddDim:.*]] = rock.transform %[[valuesGQA]] {{.*}} : memref<2x1024x32xf32> to memref<2x1x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[valuesBroadcast:.*]] = rock.transform %[[valuesAddDim]] {{.*}} : memref<2x1x1024x32xf32> to memref<2x2x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[values:.*]] = rock.transform %[[valuesBroadcast]] {{.*}} : memref<2x2x1024x32xf32> to memref<4x1024x32xf32>

// CHECK_NO_SCALE-NEXT: rock.attention
// CHECK_NO_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_NO_SCALE: %[[output]] = softmax(qk) * %[[values]]
// CHECK_NO_SCALE: return

// CHECK_NO_SCALE-LABEL: func.func @host_naive_attention
// CHECK_NO_SCALE: %[[keysExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 32, 1024] : tensor<2x32x1024xf32> into tensor<2x1x32x1024xf32>
// CHECK_NO_SCALE: %[[keysAdd:.*]] = tosa.add %{{.*}}, %[[keysExpanded]] : (tensor<2x2x32x1024xf32>, tensor<2x1x32x1024xf32>) -> tensor<2x2x32x1024xf32>
// CHECK_NO_SCALE: %[[keysTensor:.*]] = tensor.collapse_shape %[[keysAdd]] {{.*}} : tensor<2x2x32x1024xf32> into tensor<4x32x1024xf32>
// CHECK_NO_SCALE: %[[valuesExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 1024, 32] : tensor<2x1024x32xf32> into tensor<2x1x1024x32xf32>
// CHECK_NO_SCALE: %[[valuesAdd:.*]] = tosa.add %{{.*}}, %[[valuesExpanded]] : (tensor<2x2x1024x32xf32>, tensor<2x1x1024x32xf32>) -> tensor<2x2x1024x32xf32>
// CHECK_NO_SCALE: %[[valuesTensor:.*]] = tensor.collapse_shape %[[valuesAdd]] {{.*}} : tensor<2x2x1024x32xf32> into tensor<4x1024x32xf32>
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
