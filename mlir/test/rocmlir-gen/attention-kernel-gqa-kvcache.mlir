// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -current_seq_len=33 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_SCALE

// CHECK_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_SCALE-LABEL: func.func @rock_attention
// CHECK_SCALE-SAME: (%[[queriesRaw:.*0]]: memref<131072xf32>,
// CHECK_SCALE-SAME: %[[keysRaw:.*1]]: memref<65536xf32>,
// CHECK_SCALE-SAME: %[[valuesRaw:.*2]]: memref<65536xf32>,
// CHECK_SCALE-SAME: %[[scaleRaw:.*3]]: memref<4194304xf32>,
// CHECK_SCALE-SAME: %[[currentSeqLenRaw:.*4]]: memref<1xi32>,
// CHECK_SCALE-SAME: %[[outputRaw:.*5]]: memref<131072xf32>)
// CHECK_SCALE-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK_SCALE-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK_SCALE-NEXT: %[[keysGQA:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<65536xf32> to memref<2x32x1024xf32>
// CHECK_SCALE-NEXT: %[[valuesGQA:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<65536xf32> to memref<2x1024x32xf32>
// CHECK_SCALE-NEXT: %[[scale:.*]] = rock.transform %[[scaleRaw]] {{.*}} : memref<4194304xf32> to memref<4x1024x1024xf32>
// CHECK_SCALE-NEXT: %[[currentSeqLen:.*]] = rock.transform %[[currentSeqLenRaw]] {{.*}} : memref<1xi32> to memref<1xi32>
// CHECK_SCALE-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK_SCALE-NEXT: %[[currentSeqLenAddDim:.*]] = rock.transform %[[currentSeqLen]] {{.*}} : memref<1xi32> to memref<1x1xi32>
// CHECK_SCALE-NEXT: %[[currentSeqLenBroadcast:.*]] = rock.transform %[[currentSeqLenAddDim]] {{.*}} : memref<1x1xi32> to memref<1x4xi32>
// CHECK_SCALE-NEXT: %[[currentSeqLenMerge:.*]] = rock.transform %[[currentSeqLenBroadcast]] {{.*}} : memref<1x4xi32> to memref<4xi32>
// CHECK_SCALE-NEXT: %[[keysAddDim:.*]] = rock.transform %[[keysGQA]] {{.*}} : memref<2x32x1024xf32> to memref<2x1x32x1024xf32>
// CHECK_SCALE-NEXT: %[[keysBroadcast:.*]] = rock.transform %[[keysAddDim]] {{.*}} : memref<2x1x32x1024xf32> to memref<2x2x32x1024xf32>
// CHECK_SCALE-NEXT: %[[keys:.*]] = rock.transform %[[keysBroadcast]] {{.*}} : memref<2x2x32x1024xf32> to memref<4x32x1024xf32>
// CHECK_SCALE-NEXT: %[[valuesAddDim:.*]] = rock.transform %[[valuesGQA]] {{.*}} : memref<2x1024x32xf32> to memref<2x1x1024x32xf32>
// CHECK_SCALE-NEXT: %[[valuesBroadcast:.*]] = rock.transform %[[valuesAddDim]] {{.*}} : memref<2x1x1024x32xf32> to memref<2x2x1024x32xf32>
// CHECK_SCALE-NEXT: %[[values:.*]] = rock.transform %[[valuesBroadcast]] {{.*}} : memref<2x2x1024x32xf32> to memref<4x1024x32xf32>

// CHECK_SCALE-NEXT: rock.attention
// CHECK_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_SCALE-NEXT: currentSeqLen = (%[[currentSeqLenMerge]] : memref<4xi32>)
// CHECK_SCALE-NEXT: qk = elementwise otherIns(%[[scale]]
// CHECK_SCALE: %[[output]] = softmax(qk) * %[[values]]
// CHECK_SCALE: return

// CHECK_SCALE-LABEL: func.func @host_naive_attention
// CHECK_SCALE: %[[keysExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 32, 1024] : tensor<2x32x1024xf32> into tensor<2x1x32x1024xf32>
// CHECK_SCALE: %[[keysAdd:.*]] = tosa.add %{{.*}}, %[[keysExpanded]] : (tensor<2x2x32x1024xf32>, tensor<2x1x32x1024xf32>) -> tensor<2x2x32x1024xf32>
// CHECK_SCALE: %[[keysTensor:.*]] = tensor.collapse_shape %[[keysAdd]] {{.*}} : tensor<2x2x32x1024xf32> into tensor<4x32x1024xf32>
// CHECK_SCALE: %[[valuesExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 1024, 32] : tensor<2x1024x32xf32> into tensor<2x1x1024x32xf32>
// CHECK_SCALE: %[[valuesAdd:.*]] = tosa.add %{{.*}}, %[[valuesExpanded]] : (tensor<2x2x1024x32xf32>, tensor<2x1x1024x32xf32>) -> tensor<2x2x1024x32xf32>
// CHECK_SCALE: %[[valuesTensor:.*]] = tensor.collapse_shape %[[valuesAdd]] {{.*}} : tensor<2x2x1024x32xf32> into [[valuesShape:tensor<.*>]]
// CHECK_SCALE: %[[qkTensorOrig:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor]], %{{.*}}, %{{.*}} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]

// CHECK_SCALE: %[[currSeqLenTensorDumbReshaped:.*]] = tosa.reshape %[[currSeqLenTensor:.*]], %{{.*}} : (tensor<1xi32>, !tosa.shape<1>) -> tensor<1xi32>
// CHECK_SCALE: %[[currSeqLenTensorReshaped:.*]] = tosa.reshape %[[currSeqLenTensorDumbReshaped]], %{{.*}} : (tensor<1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>

// CHECK_SCALE: %[[scaledFirstReshaped:.*]] = tosa.reshape %[[scaledTensorRaw:.*]], %{{.*}} : (tensor<4194304xf32>, !tosa.shape<3>) -> tensor<4x1024x1024xf32>
// CHECK_SCALE: %[[scaledReshaped:.*]] = tosa.reshape %[[scaledFirstReshaped:.*]], %{{.*}} : (tensor<4x1024x1024xf32>, !tosa.shape<4>) -> tensor<1x4x1024x1024xf32>
// CHECK_SCALE: %[[range2:.*]] = "tosa.const"() <{values = {{.*}} : tensor<1024xi32>}> : () -> tensor<1024xi32>
// CHECK_SCALE: %[[range2Reshaped:.*]] = tosa.reshape %[[range2:.*]], %{{.*}} : (tensor<1024xi32>, !tosa.shape<4>) -> tensor<1x1x1x1024xi32>
// CHECK_SCALE: %[[zero:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x1024x1024xi32>}> : () -> tensor<1x4x1024x1024xi32>
// CHECK_SCALE: %[[rangeBroadcast2:.*]] = tosa.add %[[zero]], %[[range2Reshaped]] : (tensor<1x4x1024x1024xi32>, tensor<1x1x1x1024xi32>) -> tensor<1x4x1024x1024xi32>
// CHECK_SCALE: %[[zero2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x1024x1024xi32>}> : () -> tensor<1x4x1024x1024xi32>
// CHECK_SCALE: %[[currSeqLenTensorBroadcast2:.*]] = tosa.add %[[zero2]], %[[currSeqLenTensorReshaped]] : (tensor<1x4x1024x1024xi32>, tensor<1x1x1x1xi32>) -> tensor<1x4x1024x1024xi32>
// CHECK_SCALE: %[[mask2:.*]] = tosa.greater %[[rangeBroadcast2]], %[[currSeqLenTensorBroadcast2]] : (tensor<1x4x1024x1024xi32>, tensor<1x4x1024x1024xi32>) -> tensor<1x4x1024x1024xi1>
// CHECK_SCALE: %[[one:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x4x1024x1024xf32>}> : () -> tensor<1x4x1024x1024xf32>
// CHECK_SCALE: %[[scaleTensorBeforeReshape:.*]] = tosa.select %[[mask2]], %[[one]], %[[scaledReshaped]] : (tensor<1x4x1024x1024xi1>, tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
// CHECK_SCALE: %[[scaleTensor:.*]] = tosa.reshape %[[scaleTensorBeforeReshape]], %{{.*}} : (tensor<1x4x1024x1024xf32>, !tosa.shape<3>) -> tensor<4x1024x1024xf32>
// CHECK_SCALE: %[[sqkTensor:.*]] = tosa.mul %[[qkTensorOrig]], %[[scaleTensor]], %{{.*}} : ([[squareShape]], [[squareShape]], tensor<1xi8>) -> [[squareShape]]

// CHECK_SCALE: %[[sqkTensorCast:.*]] = tosa.cast %[[sqkTensor]] : (tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
// CHECK_SCALE: %[[qkTensorReshaped:.*]] = tosa.reshape %[[sqkTensorCast]], %{{.*}} : (tensor<4x1024x1024xf32>, !tosa.shape<4>) -> tensor<1x4x1024x1024xf32>
// CHECK_SCALE: %[[range:.*]] = "tosa.const"() <{values = {{.*}} : tensor<1024xi32>}> : () -> tensor<1024xi32>
// CHECK_SCALE: %[[rangeReshaped:.*]] = tosa.reshape %[[range:.*]], %{{.*}} : (tensor<1024xi32>, !tosa.shape<4>) -> tensor<1x1x1x1024xi32>
// CHECK_SCALE: %[[zero3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x1024x1024xi32>}> : () -> tensor<1x4x1024x1024xi32>
// CHECK_SCALE: %[[rangeBroadcast:.*]] = tosa.add %[[zero3]], %[[rangeReshaped]] : (tensor<1x4x1024x1024xi32>, tensor<1x1x1x1024xi32>) -> tensor<1x4x1024x1024xi32>
// CHECK_SCALE: %[[zero4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x1024x1024xi32>}> : () -> tensor<1x4x1024x1024xi32>
// CHECK_SCALE: %[[currSeqLenTensorBroadcast:.*]] = tosa.add %[[zero4]], %[[currSeqLenTensorReshaped]] : (tensor<1x4x1024x1024xi32>, tensor<1x1x1x1xi32>) -> tensor<1x4x1024x1024xi32>
// CHECK_SCALE: %[[mask:.*]] = tosa.greater %[[rangeBroadcast]], %[[currSeqLenTensorBroadcast]] : (tensor<1x4x1024x1024xi32>, tensor<1x4x1024x1024xi32>) -> tensor<1x4x1024x1024xi1>
// CHECK_SCALE: %[[negInf:.*]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x4x1024x1024xf32>}> : () -> tensor<1x4x1024x1024xf32>
// CHECK_SCALE: %[[qkTensorBeforeReshape:.*]] = tosa.select %[[mask]], %[[negInf]], %[[qkTensorReshaped]] : (tensor<1x4x1024x1024xi1>, tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
// CHECK_SCALE: %[[qkTensor:.*]] = tosa.reshape %[[qkTensorBeforeReshape]], %{{.*}} : (tensor<1x4x1024x1024xf32>, !tosa.shape<3>) -> tensor<4x1024x1024xf32>

// CHECK_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[qkTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_SCALE-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[qkTensor]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor]], %{{.*}}, %{{.*}} : ([[squareShape]], [[valuesShape]], tensor<1xf32>, tensor<1xf32>) -> tensor<4x1024x32xf32>
// CHECK_SCALE: return

// ----

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -current_seq_len=33 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_NO_SCALE

// CHECK_NO_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_NO_SCALE-LABEL: func.func @rock_attention
// CHECK_NO_SCALE-SAME: (%[[queriesRaw:.*0]]: memref<131072xf32>,
// CHECK_NO_SCALE-SAME: %[[keysRaw:.*1]]: memref<65536xf32>,
// CHECK_NO_SCALE-SAME: %[[valuesRaw:.*2]]: memref<65536xf32>,
// CHECK_NO_SCALE-SAME: %[[currentSeqLenRaw:.*3]]: memref<1xi32>,
// CHECK_NO_SCALE-SAME: %[[outputRaw:.*4]]: memref<131072xf32>)
// CHECK_NO_SCALE-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK_NO_SCALE-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[keysGQA:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<65536xf32> to memref<2x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[valuesGQA:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<65536xf32> to memref<2x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[currentSeqLen:.*]] = rock.transform %[[currentSeqLenRaw]] {{.*}} : memref<1xi32> to memref<1xi32>
// CHECK_NO_SCALE-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<131072xf32> to memref<4x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[currentSeqLenAddDim:.*]] = rock.transform %[[currentSeqLen]] {{.*}} : memref<1xi32> to memref<1x1xi32>
// CHECK_NO_SCALE-NEXT: %[[currentSeqLenBroadcast:.*]] = rock.transform %[[currentSeqLenAddDim]] {{.*}} : memref<1x1xi32> to memref<1x4xi32>
// CHECK_NO_SCALE-NEXT: %[[currentSeqLenMerge:.*]] = rock.transform %[[currentSeqLenBroadcast]] {{.*}} : memref<1x4xi32> to memref<4xi32>
// CHECK_NO_SCALE-NEXT: %[[keysAddDim:.*]] = rock.transform %[[keysGQA]] {{.*}} : memref<2x32x1024xf32> to memref<2x1x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[keysBroadcast:.*]] = rock.transform %[[keysAddDim]] {{.*}} : memref<2x1x32x1024xf32> to memref<2x2x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[keys:.*]] = rock.transform %[[keysBroadcast]] {{.*}} : memref<2x2x32x1024xf32> to memref<4x32x1024xf32>
// CHECK_NO_SCALE-NEXT: %[[valuesAddDim:.*]] = rock.transform %[[valuesGQA]] {{.*}} : memref<2x1024x32xf32> to memref<2x1x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[valuesBroadcast:.*]] = rock.transform %[[valuesAddDim]] {{.*}} : memref<2x1x1024x32xf32> to memref<2x2x1024x32xf32>
// CHECK_NO_SCALE-NEXT: %[[values:.*]] = rock.transform %[[valuesBroadcast]] {{.*}} : memref<2x2x1024x32xf32> to memref<4x1024x32xf32>

// CHECK_NO_SCALE-NEXT: rock.attention
// CHECK_NO_SCALE-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK_NO_SCALE-NEXT: currentSeqLen = (%[[currentSeqLenMerge]] : memref<4xi32>)
// CHECK_NO_SCALE: %[[output]] = softmax(qk) * %[[values]]
// CHECK_NO_SCALE: return

// CHECK_NO_SCALE-LABEL: func.func @host_naive_attention
// CHECK_NO_SCALE: %[[keysExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 32, 1024] : tensor<2x32x1024xf32> into tensor<2x1x32x1024xf32>
// CHECK_NO_SCALE: %[[keysAdd:.*]] = tosa.add %{{.*}}, %[[keysExpanded]] : (tensor<2x2x32x1024xf32>, tensor<2x1x32x1024xf32>) -> tensor<2x2x32x1024xf32>
// CHECK_NO_SCALE: %[[keysTensor:.*]] = tensor.collapse_shape %[[keysAdd]] {{.*}} : tensor<2x2x32x1024xf32> into tensor<4x32x1024xf32>
// CHECK_NO_SCALE: %[[valuesExpanded:.*]] = tensor.expand_shape {{.*}} output_shape [2, 1, 1024, 32] : tensor<2x1024x32xf32> into tensor<2x1x1024x32xf32>
// CHECK_NO_SCALE: %[[valuesAdd:.*]] = tosa.add %{{.*}}, %[[valuesExpanded]] : (tensor<2x2x1024x32xf32>, tensor<2x1x1024x32xf32>) -> tensor<2x2x1024x32xf32>
// CHECK_NO_SCALE: %[[valuesTensor:.*]] = tensor.collapse_shape %[[valuesAdd]] {{.*}} : tensor<2x2x1024x32xf32> into [[valuesShape:tensor<.*>]]
// CHECK_NO_SCALE: %[[qkTensorOrig:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]

// CHECK_NO_SCALE: %[[currSeqLenTensorDumbReshaped:.*]] = tosa.reshape %[[currSeqLenTensor:.*]], %{{.*}} : (tensor<1xi32>, !tosa.shape<1>) -> tensor<1xi32>
// CHECK_NO_SCALE: %[[currSeqLenTensorReshaped:.*]] = tosa.reshape %[[currSeqLenTensorDumbReshaped]], %{{.*}} : (tensor<1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
// CHECK_NO_SCALE: %[[qkTensorCast:.*]] = tosa.cast %[[qkTensorOrig]] : (tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
// CHECK_NO_SCALE: %[[qkTensorReshaped:.*]] = tosa.reshape %[[qkTensorCast]], %{{.*}} : (tensor<4x1024x1024xf32>, !tosa.shape<4>) -> tensor<1x4x1024x1024xf32>
// CHECK_NO_SCALE: %[[range:.*]] = "tosa.const"() <{values = {{.*}} : tensor<1024xi32>}> : () -> tensor<1024xi32>
// CHECK_NO_SCALE: %[[rangeReshaped:.*]] = tosa.reshape %[[range]], %{{.*}} : (tensor<1024xi32>, !tosa.shape<4>) -> tensor<1x1x1x1024xi32>
// CHECK_NO_SCALE: %[[zero:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x1024x1024xi32>}> : () -> tensor<1x4x1024x1024xi32>
// CHECK_NO_SCALE: %[[rangeBroadcast:.*]] = tosa.add %[[zero]], %[[rangeReshaped]] : (tensor<1x4x1024x1024xi32>, tensor<1x1x1x1024xi32>) -> tensor<1x4x1024x1024xi32>
// CHECK_NO_SCALE: %[[zero2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x4x1024x1024xi32>}> : () -> tensor<1x4x1024x1024xi32>
// CHECK_NO_SCALE: %[[currSeqLenTensorBroadcast:.*]] = tosa.add %[[zero2]], %[[currSeqLenTensorReshaped]] : (tensor<1x4x1024x1024xi32>, tensor<1x1x1x1xi32>) -> tensor<1x4x1024x1024xi32>
// CHECK_NO_SCALE: %[[mask:.*]] = tosa.greater %[[rangeBroadcast]], %[[currSeqLenTensorBroadcast]] : (tensor<1x4x1024x1024xi32>, tensor<1x4x1024x1024xi32>) -> tensor<1x4x1024x1024xi1>
// CHECK_NO_SCALE: %[[negInf:.*]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x4x1024x1024xf32>}> : () -> tensor<1x4x1024x1024xf32>
// CHECK_NO_SCALE: %[[qkTensorBeforeReshape:.*]] = tosa.select %[[mask]], %[[negInf]], %[[qkTensorReshaped]] : (tensor<1x4x1024x1024xi1>, tensor<1x4x1024x1024xf32>, tensor<1x4x1024x1024xf32>) -> tensor<1x4x1024x1024xf32>
// CHECK_NO_SCALE: %[[qkTensor:.*]] = tosa.reshape %[[qkTensorBeforeReshape]], %{{.*}} : (tensor<1x4x1024x1024xf32>, !tosa.shape<3>) -> tensor<4x1024x1024xf32>

// CHECK_NO_SCALE-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[qkTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape:tensor<.*>]]
// CHECK_NO_SCALE-DAG: %[[normilizedQkTensor:.*]] = tosa.sub %[[qkTensor]], %[[sqkMaxs]] : ([[squareShape]], [[reducedShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedQkTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShape]]) -> [[reducedShape]]
// CHECK_NO_SCALE-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : ([[reducedShape]]) -> [[reducedShape]]
// CHECK_NO_SCALE-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShape]], [[reducedShape]], tensor<1xi8>) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShape]]) -> [[squareShape]]
// CHECK_NO_SCALE-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> tensor<4x1024x32xf32>
// CHECK_NO_SCALE: return
