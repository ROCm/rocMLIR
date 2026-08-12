// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f16 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f16 -pv --apply-bufferization-pipeline=false --schedule_version 2 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=SCHEDV2
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f16 -pv --apply-bufferization-pipeline=false --schedule_version 3 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=SCHEDV3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 --with-attn-scale -t f16 -pv --apply-bufferization-pipeline=false --schedule_version 4 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=SCHEDV4

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// SCHEDV2-LABEL: func.func @rock_attention
// SCHEDV2-SAME: rock.schedule_version = #rock.rock.schedule_version<2>

// SCHEDV3-LABEL: func.func @rock_attention
// SCHEDV3-SAME: rock.schedule_version = #rock.rock.schedule_version<3>

// SCHEDV4-LABEL: func.func @rock_attention
// SCHEDV4-SAME: rock.schedule_version = #rock.rock.schedule_version<4>

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: memref<32768xf16>,
// CHECK-SAME: %[[keysRaw:.*1]]: memref<32768xf16>,
// CHECK-SAME: %[[valuesRaw:.*2]]: memref<32768xf16>,
// CHECK-SAME: %[[scaleRaw:.*3]]: memref<1048576xf16>,
// CHECK-SAME: %[[outputRaw:.*4]]: memref<32768xf16>)
// CHECK-SAME: attributes {mhal.arch = "[[$ARCH]]", rock.kernel}
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
// CHECK-DAG: %[[normilizedSqkTensor:.*]] = tosa.sub %[[sqkTensorCast]], %{{.*}} : ([[squareShapeF32]], [[reducedShape]]) -> [[squareShapeF32]]
// CHECK-DAG: %[[expsTensor:.*]] = tosa.exp %[[normilizedSqkTensor]] : ([[squareShapeF32]]) -> [[squareShapeF32]]
// CHECK-DAG: %[[expsSumsTensor:.*]] = tosa.reduce_sum %[[expsTensor]] {{.*}} : ([[squareShapeF32]]) -> [[reducedShape]]
// CHECK-DAG: %[[invExpsSums:.*]] = tosa.reciprocal %{{.*}} : ([[reducedShape]]) -> [[reducedShape]]
// CHECK-DAG: %[[softmaxTensor:.*]] = tosa.mul %[[expsTensor]], %[[invExpsSums]], %{{.*}} : ([[squareShapeF32]], [[reducedShape]], tensor<1xi8>) -> [[squareShapeF32]] 
// CHECK-DAG: %[[softmaxTensorCast:.*]] = tosa.cast %[[softmaxTensor]] : ([[squareShapeF32]]) -> [[squareShape]]
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[softmaxTensorCast]], %[[valuesTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[squareShape]], [[valuesShape:tensor<.*>]], tensor<1xf16>, tensor<1xf16>) -> [[squareShape:tensor<.*>]]
// CHECK: return

// `--softmax_dtype` controls the type used for the softmax intermediate
// inside `rock.attention`. The default for an f16 input kernel is f32
// (wider than the operand type for numerical stability); requesting f16
// collapses the intermediate to the operand type to save footprint.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 | FileCheck %s --check-prefix=SOFTMAX_DEFAULT_F16
// SOFTMAX_DEFAULT_F16: rock.attention
// SOFTMAX_DEFAULT_F16: softmaxType = f32
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 --softmax_dtype f16 | FileCheck %s --check-prefix=SOFTMAX_F16
// SOFTMAX_F16: rock.attention
// SOFTMAX_F16: softmaxType = f16
