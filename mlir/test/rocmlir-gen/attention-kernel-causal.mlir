// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -causal -seq_len_q 512 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: memref<16384xf32>,
// CHECK-SAME: %[[keysRaw:.*1]]: memref<32768xf32>,
// CHECK-SAME: %[[valuesRaw:.*2]]: memref<32768xf32>,
// CHECK-SAME: %[[scaleRaw:.*3]]: memref<524288xf32>,
// CHECK-SAME: %[[outputRaw:.*4]]: memref<16384xf32>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK-NEXT: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : memref<16384xf32> to memref<1x512x32xf32>
// CHECK-NEXT: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : memref<32768xf32> to memref<1x32x1024xf32>
// CHECK-NEXT: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>
// CHECK-NEXT: %[[scale:.*]] = rock.transform %[[scaleRaw]] {{.*}} : memref<524288xf32> to memref<1x512x1024xf32>
// CHECK-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<16384xf32> to memref<1x512x32xf32>

// CHECK-NEXT: rock.attention
// CHECK-NEXT: qk = %[[queries]] * %[[keys]]
// CHECK-NEXT: causal
// CHECK-NEXT: qk = elementwise otherIns(%[[scale]]
// CHECK: %[[output]] = softmax(qk) * %[[values]]
// CHECK: return

// CHECK-LABEL: func.func @host_naive_attention
// CHECK: %[[qkTensorOrig:.*]] = tosa.matmul %[[queriesTensor:.*]], %[[keysTensor:.*]], %{{.*}}, %{{.*}} : ([[queriesShape:tensor<.*>]], [[keysShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]

// CHECK: %[[scaledReshaped:.*]] = tosa.reshape %[[scaledTensorRaw:.*]], %{{.*}} : (tensor<524288xf32>, !tosa.shape<3>) -> tensor<1x512x1024xf32>
// CHECK: %[[range2:.*]] = "tosa.const"() <{values = {{.*}} : tensor<512xi32>}> : () -> tensor<512xi32>
// CHECK: %[[range2Reshaped:.*]] = tosa.reshape %[[range2:.*]], %{{.*}} : (tensor<512xi32>, !tosa.shape<3>) -> tensor<1x512x1xi32>
// CHECK: %[[zero:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x512x1024xi32>}> : () -> tensor<1x512x1024xi32>
// CHECK: %[[rangeBroadcast2:.*]] = tosa.add %[[zero]], %[[range2Reshaped]] : (tensor<1x512x1024xi32>, tensor<1x512x1xi32>) -> tensor<1x512x1024xi32>
// CHECK: %[[range3:.*]] = "tosa.const"() <{values = {{.*}} : tensor<1024xi32>}> : () -> tensor<1024xi32>
// CHECK: %[[range3Reshaped:.*]] = tosa.reshape %[[range3:.*]], %{{.*}} : (tensor<1024xi32>, !tosa.shape<3>) -> tensor<1x1x1024xi32>
// CHECK: %[[zero2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x512x1024xi32>}> : () -> tensor<1x512x1024xi32>
// CHECK: %[[rangeBroadcast3:.*]] = tosa.add %[[zero2]], %[[range3Reshaped]] : (tensor<1x512x1024xi32>, tensor<1x1x1024xi32>) -> tensor<1x512x1024xi32>
// CHECK: %[[mask2:.*]] = tosa.greater %[[rangeBroadcast3]], %[[rangeBroadcast2]] : (tensor<1x512x1024xi32>, tensor<1x512x1024xi32>) -> tensor<1x512x1024xi1>
// CHECK: %[[ones:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x512x1024xf32>}> : () -> tensor<1x512x1024xf32>
// CHECK: %[[scaleTensor:.*]] = tosa.select %[[mask2]], %[[ones]], %[[scaledReshaped]] : (tensor<1x512x1024xi1>, tensor<1x512x1024xf32>, tensor<1x512x1024xf32>) -> tensor<1x512x1024xf32>
// CHECK: %[[sqkTensor:.*]] = tosa.mul %[[qkTensorOrig]], %[[scaleTensor]], %{{.*}} : (tensor<1x512x1024xf32>, tensor<1x512x1024xf32>, tensor<1xi8>) -> tensor<1x512x1024xf32>

// CHECK: %[[range4:.*]] = "tosa.const"() <{values = {{.*}} : tensor<512xi32>}> : () -> tensor<512xi32>
// CHECK: %[[range4Reshaped:.*]] = tosa.reshape %[[range4:.*]], %{{.*}} : (tensor<512xi32>, !tosa.shape<3>) -> tensor<1x512x1xi32>
// CHECK: %[[zero3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x512x1024xi32>}> : () -> tensor<1x512x1024xi32>
// CHECK: %[[rangeBroadcast4:.*]] = tosa.add %[[zero3]], %[[range4Reshaped]] : (tensor<1x512x1024xi32>, tensor<1x512x1xi32>) -> tensor<1x512x1024xi32>
// CHECK: %[[range5:.*]] = "tosa.const"() <{values = {{.*}} : tensor<1024xi32>}> : () -> tensor<1024xi32>
// CHECK: %[[range5Reshaped:.*]] = tosa.reshape %[[range5:.*]], %{{.*}} : (tensor<1024xi32>, !tosa.shape<3>) -> tensor<1x1x1024xi32>
// CHECK: %[[zero4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x512x1024xi32>}> : () -> tensor<1x512x1024xi32>
// CHECK: %[[rangeBroadcast5:.*]] = tosa.add %[[zero4]], %[[range5Reshaped]] : (tensor<1x512x1024xi32>, tensor<1x1x1024xi32>) -> tensor<1x512x1024xi32>
// CHECK: %[[mask3:.*]] = tosa.greater %[[rangeBroadcast5]], %[[rangeBroadcast4]] : (tensor<1x512x1024xi32>, tensor<1x512x1024xi32>) -> tensor<1x512x1024xi1>
// CHECK: %[[negInf:.*]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x512x1024xf32>}> : () -> tensor<1x512x1024xf32>
// CHECK: %[[qkTensor:.*]] = tosa.select %[[mask3]], %[[negInf]], %[[sqkTensor]] : (tensor<1x512x1024xi1>, tensor<1x512x1024xf32>, tensor<1x512x1024xf32>) -> tensor<1x512x1024xf32

// CHECK-DAG: %[[sqkMaxs:.*]] = tosa.reduce_max %[[qkTensor]] {{.*}} : (tensor<1x512x1024xf32>) -> tensor<1x512x1xf32>
// CHECK-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[qkTensor]], %[[sqkMaxs]] : (tensor<1x512x1024xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1024xf32>
// CHECK-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : (tensor<1x512x1024xf32>) -> tensor<1x512x1024xf32>
// CHECK-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : (tensor<1x512x1024xf32>) -> tensor<1x512x1xf32>
// CHECK-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %[[expsSumsTensor]] : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
// CHECK-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : (tensor<1x512x1024xf32>, tensor<1x512x1xf32>, tensor<1xi8>) -> tensor<1x512x1024xf32>
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensor]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} : (tensor<1x512x1024xf32>, tensor<1x1024x32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x512x32xf32>
// CHECK: return