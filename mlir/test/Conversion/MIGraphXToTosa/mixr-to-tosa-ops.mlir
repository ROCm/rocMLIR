// RUN: rocmlir-opt -split-input-file --migraphx-transform --canonicalize --migraphx-to-tosa %s -verify-diagnostics -o -| FileCheck %s

module  {
  // CHECK-LABEL: func @dequantize_scale
  // CHECK-NOT: tosa.sub
  // CHECK: tosa.mul
  func.func @dequantize_scale(%arg: tensor<1x112x112x64xi32>, %scale: tensor<64xf32>) -> tensor<1x112x112x64xf32> attributes {kernel = "mixr"} {
    %1 = "migraphx.dequantizelinear"(%arg, %scale) : (tensor<1x112x112x64xi32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
    return %1 : tensor<1x112x112x64xf32>
  }

  // CHECK-LABEL: func @dequantize_scale_bias
  // CHECK: tosa.sub
  // CHECK: tosa.mul
  func.func @dequantize_scale_bias(%arg: tensor<1x112x112x64xi32>, %scale: tensor<64xf32>, %bias: tensor<64xi32>) -> tensor<1x112x112x64xf32> attributes {kernel = "mixr"} {
    %1 = "migraphx.dequantizelinear"(%arg, %scale, %bias) : (tensor<1x112x112x64xi32>, tensor<64xf32>, tensor<64xi32>) -> tensor<1x112x112x64xf32>
    return %1 : tensor<1x112x112x64xf32>
  }

  // CHECK-LABEL: func @quantize_scale
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK-NOT: tosa.add
  func.func @quantize_scale(%arg: tensor<1x112x112x64xf32>, %scale: tensor<64xf32>) -> tensor<1x112x112x64xi8> attributes {kernel = "mixr"} {
    %1 = "migraphx.quantizelinear"(%arg, %scale) : (tensor<1x112x112x64xf32>, tensor<64xf32>) -> tensor<1x112x112x64xi8>
    return %1 : tensor<1x112x112x64xi8>
}

  // CHECK-LABEL: func @quantize_scale_bias
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.add
  func.func @quantize_scale_bias(%arg: tensor<1x112x112x64xf32>, %scale: tensor<64xf32>, %bias: tensor<64xi8>) -> tensor<1x112x112x64xi8> attributes {kernel = "mixr"} {
    %1 = "migraphx.quantizelinear"(%arg, %scale, %bias) : (tensor<1x112x112x64xf32>, tensor<64xf32>, tensor<64xi8>) -> tensor<1x112x112x64xi8>
    return %1 : tensor<1x112x112x64xi8>
  }

  // CHECK-LABEL: func @conv_with_quant
  // CHECK: tosa.conv2d{{.*}} quantization_info
  // CHECK: tosa.sub
  // CHECK: tosa.mul
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.add
  func.func @conv_with_quant(%arg1: tensor<1x3x224x224xi8>, %arg2: tensor<64x3x7x7xi8>, %scale: tensor<1x64x1x1xf32>, %bias: tensor<1x64x1x1xi32>, %bias2: tensor<1x64x1x1xi8>) -> tensor<1x64x112x112xi8> attributes {kernel = "mixr"} {
    %1 = migraphx.quant_convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x3x224x224xi8>, tensor<64x3x7x7xi8>) -> tensor<1x64x112x112xi32>
    %2 = "migraphx.dequantizelinear"(%1, %scale, %bias) : (tensor<1x64x112x112xi32>, tensor<1x64x1x1xf32>, tensor<1x64x1x1xi32>) -> tensor<1x64x112x112xf32>
    %3 = "migraphx.quantizelinear"(%2, %scale, %bias2) : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>, tensor<1x64x1x1xi8>) -> tensor<1x64x112x112xi8>
    return %3 : tensor<1x64x112x112xi8>
  }

  // CHECK-LABEL: func.func @matmul
  // CHECK-NOT: tosa.reshape
  // CHECK: tosa.matmul
  // CHECK-NOT: tosa.reshape
  func.func @matmul(%arg0: tensor<2x256x384xf32>, %arg1: tensor<2x384x768xf32>) -> tensor<2x256x768xf32> {
    %0 = migraphx.dot(%arg0, %arg1) : (tensor<2x256x384xf32>, tensor<2x384x768xf32>) -> tensor<2x256x768xf32>
     return %0 : tensor<2x256x768xf32>
  }

  // CHECK-LABEL: func.func @quant_matmul
  // CHECK: tosa.matmul
  func.func @quant_matmul(%arg0: tensor<2x256x384xi8>, %arg1: tensor<2x384x768xi8>) -> tensor<2x256x768xi32> {
    %0 = migraphx.quant_dot(%arg0, %arg1) : (tensor<2x256x384xi8>, tensor<2x384x768xi8>) -> tensor<2x256x768xi32>
     return %0 : tensor<2x256x768xi32>
  }

  // CHECK-LABEL: func.func @matmul_larger_batch
  // CHECK: tosa.matmul
  func.func @matmul_larger_batch(%arg0: tensor<2x16x256x384xf32>, %arg1: tensor<2x16x384x768xf32>) -> tensor<2x16x256x768xf32> {
    %0 = migraphx.dot(%arg0, %arg1) : (tensor<2x16x256x384xf32>, tensor<2x16x384x768xf32>) -> tensor<2x16x256x768xf32>
     return %0 : tensor<2x16x256x768xf32>
  }

  // CHECK-LABEL: func.func @matmul_rank2
  // CHECK: tosa.matmul
  func.func @matmul_rank2(%arg0: tensor<32x72xf32>, %arg1: tensor<72x64xf32>) -> tensor<32x64xf32> {
    %0 = migraphx.dot(%arg0, %arg1) : (tensor<32x72xf32>, tensor<72x64xf32>) -> tensor<32x64xf32>
     return %0 : tensor<32x64xf32>
  }

  // CHECK-LABEL: func.func @matmul_broadcast
  func.func @matmul_broadcast(%arg0: tensor<64x64x2304xf16>, %arg1: tensor<64x64x768xf16>, %arg2: tensor<1x768x2304xf16>) -> tensor<64x64x2304xf16> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    %0 = migraphx.multibroadcast(%arg2) {out_dyn_dims = [], out_lens = [64, 768, 2304]} : (tensor<1x768x2304xf16>) -> tensor<64x768x2304xf16>
    // CHECK-DAG: %[[RESHAPE0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 4096, 768>}>
    // CHECK-DAG: %[[RESHAPE1:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 768, 2304>}>
    %1 = migraphx.dot(%arg1, %0) : (tensor<64x64x768xf16>, tensor<64x768x2304xf16>) -> tensor<64x64x2304xf16>
    // CHECK-DAG: %[[MATMUL:.*]] = "tosa.matmul"(%[[RESHAPE0]], %[[RESHAPE1]]
    // CHECK: %[[RESHAPE2:.*]] = "tosa.reshape"(%[[MATMUL]]) <{new_shape = array<i64: 64, 64, 2304>}>
    %2 = migraphx.add(%1, %arg0) : (tensor<64x64x2304xf16>, tensor<64x64x2304xf16>) -> tensor<64x64x2304xf16>
    return %2 : tensor<64x64x2304xf16>
  }

  // CHECK-LABEL: func.func @matmul_broadcast_R5
  func.func @matmul_broadcast_R5(%arg0: tensor<2x4x8x64x2304xf16>, %arg1: tensor<2x4x8x64x768xf16>, %arg2: tensor<1x1x1x768x2304xf16>) -> tensor<2x4x8x64x2304xf16> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    %0 = migraphx.multibroadcast(%arg2) {out_dyn_dims = [], out_lens = [2, 4, 8, 768, 2304]} : (tensor<1x1x1x768x2304xf16>) -> tensor<2x4x8x768x2304xf16>
    // CHECK-DAG: %[[RESHAPE0:.*]] = "tosa.reshape"(%arg1) <{new_shape = array<i64: 1, 4096, 768>}>
    // CHECK-DAG: %[[RESHAPE1:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 768, 2304>}>
    %1 = migraphx.dot(%arg1, %0) : (tensor<2x4x8x64x768xf16>, tensor<2x4x8x768x2304xf16>) -> tensor<2x4x8x64x2304xf16>
    // CHECK-DAG: %[[MATMUL:.*]] = "tosa.matmul"(%[[RESHAPE0]], %[[RESHAPE1]]
    // CHECK: %[[RESHAPE2:.*]] = "tosa.reshape"(%[[MATMUL]]) <{new_shape = array<i64: 2, 4, 8, 64, 2304>}>
    %2 = migraphx.add(%1, %arg0) : (tensor<2x4x8x64x2304xf16>, tensor<2x4x8x64x2304xf16>) -> tensor<2x4x8x64x2304xf16>
    return %2 : tensor<2x4x8x64x2304xf16>
  }


  // broadcast ops will be lowered as implicit broadcast in tosa, passes if they're converted and legalize tosa.
  // CHECK-LABEL: func @func_mbcast
  func.func @func_mbcast(%arg0: tensor<1x64x1x1xf32>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> attributes {kernel = "mixr"} {
    %0 = migraphx.multibroadcast(%arg0) {out_lens = [1, 64, 112, 112]} : (tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %2 = migraphx.add(%1, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %3 = migraphx.relu(%2) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %3 : tensor<1x64x112x112xf32>
  }

  // CHECK-LABEL: func.func @clip_i32
  func.func @clip_i32(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi32>) -> tensor<64x64xi32> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK: %[[MAX:.*]] = "tosa.maximum"(%arg0, %arg1)
    // CHECK: %[[MIN:.*]] = "tosa.minimum"(%[[MAX]], %arg2)
    // CHECK: return %[[MIN]]
    %0 = migraphx.clip(%arg0, %arg1, %arg2) : (tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
    return %0 : tensor<64x64xi32>
  }

  // CHECK-LABEL: func.func @clip_broadcast
  func.func @clip_broadcast(%arg0: tensor<64x64xf16>, %arg1: tensor<1x64xf16>, %arg2: tensor<1xf16>) -> tensor<64x64xf16> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK: %[[BCAST2:.*]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1>}>
    // CHECK: %[[MAX:.*]] = "tosa.maximum"(%arg0, %arg1)
    // CHECK: %[[MIN:.*]] = "tosa.minimum"(%[[MAX]], %[[BCAST2]])
    // CHECK: return %[[MIN]]
    %0 = migraphx.multibroadcast(%arg1) {out_dyn_dims = [], out_lens = [64, 64]} : (tensor<1x64xf16>) -> tensor<64x64xf16>
    %1 = migraphx.multibroadcast(%arg2) {out_dyn_dims = [], out_lens = [64, 64]} : (tensor<1xf16>) -> tensor<64x64xf16>
    %2 = migraphx.clip(%arg0, %0, %1) : (tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
    return %2 : tensor<64x64xf16>
  }

  // CHECK-LABEL: func.func @where
  func.func @where_f32(%arg0: tensor<64x64xi8>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK: tosa.cast
    // CHECK: tosa.select
    %0 = migraphx.where(%arg0, %arg1, %arg2) : (tensor<64x64xi8>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }

  // CHECK-LABEL: func.func @where_broadcast
  func.func @where_broadcast(%arg0: tensor<64x1xi8>, %arg1: tensor<64x64xf16>, %arg2: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK: %[[CAST:.*]] = "tosa.cast"(%arg0)
    // CHECK: "tosa.select"(%[[CAST]], %arg1, %arg2)
    %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [64, 64]} : (tensor<64x1xi8>) -> tensor<64x64xi8>
    %1 = migraphx.where(%0, %arg1, %arg2) : (tensor<64x64xi8>, tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_f32
  // CHECK-SAME: (%arg0: [[INTYPE:.*]]) -> [[OUTTYPE:.*]] {
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<1.120000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK-DAG: %[[NRECIP:.*]] = "tosa.reciprocal"(%[[N]]) : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK-DAG: %[[MUL:.*]] = "tosa.mul"(%arg0, %[[NRECIP]]) <{shift = 0 : i32}> : ([[INTYPE]], tensor<1xf32>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = "tosa.reduce_sum"(%[[MUL]]) <{axis = 2 : i64}> : ([[INTYPE]]) -> [[OUTTYPE]]
  // CHECK: return %[[REDUCE_SUM]]
  func.func @func_reduce_mean_f32(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x1x112xf32> {
    %0 = "migraphx.reduce_mean"(%arg0) {axes = [2 : i64]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x1x112xf32>
    return %0 : tensor<1x64x1x112xf32>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_f16
  // CHECK-SAME: (%arg0: [[INTYPE:.*]]) -> [[OUTTYPE:.*]] {
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<1.120000e+02> : tensor<1xf16>}> : () -> tensor<1xf16>
  // CHECK-DAG: %[[NRECIP:.*]] = "tosa.reciprocal"(%[[N]]) : (tensor<1xf16>) -> tensor<1xf16>
  // CHECK-DAG: %[[MUL:.*]] = "tosa.mul"(%arg0, %[[NRECIP]]) <{shift = 0 : i32}> : ([[INTYPE]], tensor<1xf16>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = "tosa.reduce_sum"(%[[MUL]]) <{axis = 2 : i64}> : ([[INTYPE]]) -> [[OUTTYPE]]
  // CHECK: return %[[REDUCE_SUM]]
  func.func @func_reduce_mean_f16(%arg0: tensor<1x64x112x112xf16>) -> tensor<1x64x1x112xf16> {
    %0 = "migraphx.reduce_mean"(%arg0) {axes = [2 : i64]} : (tensor<1x64x112x112xf16>) -> tensor<1x64x1x112xf16>
    return %0 : tensor<1x64x1x112xf16>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i32
  // CHECK-SAME: (%arg0: [[INTYPE:.*]]) -> [[OUTTYPE:.*]] {
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[NRECIP:.*]] = "tosa.reciprocal"(%[[N]]) : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK-DAG: %[[MUL:.*]] = "tosa.mul"(%arg0, %[[NRECIP]]) <{shift = 0 : i32}> : ([[INTYPE]], tensor<1xi32>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = "tosa.reduce_sum"(%[[MUL]]) <{axis = 2 : i64}> : ([[INTYPE]]) -> [[OUTTYPE]]
  // CHECK: return %[[REDUCE_SUM]]
  func.func @func_reduce_mean_i32(%arg0: tensor<1x64x112x112xi32>) -> tensor<1x64x1x112xi32> {
    %0 = "migraphx.reduce_mean"(%arg0) {axes = [2 : i64]} : (tensor<1x64x112x112xi32>) -> tensor<1x64x1x112xi32>
    return %0 : tensor<1x64x1x112xi32>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i16
  // CHECK-SAME: (%arg0: [[INTYPE:.*]]) -> [[OUTTYPE:.*]] {
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi16>}> : () -> tensor<1xi16>
  // CHECK-DAG: %[[NRECIP:.*]] = "tosa.reciprocal"(%[[N]]) : (tensor<1xi16>) -> tensor<1xi16>
  // CHECK-DAG: %[[MUL:.*]] = "tosa.mul"(%arg0, %[[NRECIP]]) <{shift = 0 : i32}> : ([[INTYPE]], tensor<1xi16>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = "tosa.reduce_sum"(%[[MUL]]) <{axis = 2 : i64}> : ([[INTYPE]]) -> [[OUTTYPE]]
  // CHECK: return %[[REDUCE_SUM]]
  func.func @func_reduce_mean_i16(%arg0: tensor<1x64x112x112xi16>) -> tensor<1x64x1x112xi16> {
    %0 = "migraphx.reduce_mean"(%arg0) {axes = [2 : i64]} : (tensor<1x64x112x112xi16>) -> tensor<1x64x1x112xi16>
    return %0 : tensor<1x64x1x112xi16>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i8
  // CHECK-SAME: (%arg0: [[INTYPE:.*]]) -> [[OUTTYPE:.*]] {
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[NRECIP:.*]] = "tosa.reciprocal"(%[[N]]) : (tensor<1xi8>) -> tensor<1xi8>
  // CHECK-DAG: %[[MUL:.*]] = "tosa.mul"(%arg0, %[[NRECIP]]) <{shift = 0 : i32}> : ([[INTYPE]], tensor<1xi8>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = "tosa.reduce_sum"(%[[MUL]]) <{axis = 2 : i64}> : ([[INTYPE]]) -> [[OUTTYPE]]
  // CHECK: return %[[REDUCE_SUM]]
  func.func @func_reduce_mean_i8(%arg0: tensor<1x64x112x112xi8>) -> tensor<1x64x1x112xi8> {
    %0 = "migraphx.reduce_mean"(%arg0) {axes = [2 : i64]} : (tensor<1x64x112x112xi8>) -> tensor<1x64x1x112xi8>
    return %0 : tensor<1x64x1x112xi8>
  }

  // CHECK-LABEL: func.func @func_dot_mul
  // CHECK: tosa.matmul
  // CHECK: tosa.mul
  func.func @func_dot_mul(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x5x3xf32>) -> tensor<1x5x3xf32> attributes{kernel, arch = ""} {
    %0 = migraphx.dot(%arg0, %arg1) : (tensor<1x5x4xf32>, tensor<1x4x3xf32>) -> tensor<1x5x3xf32>
    %2 = "migraphx.mul"(%0, %arg2) {} : (tensor<1x5x3xf32>, tensor<1x5x3xf32>)-> tensor<1x5x3xf32>
    return %2 : tensor<1x5x3xf32>
  }

  // CHECK-LABEL: func.func @func_slice1
  // CHECK: tosa.slice
  func.func @func_slice1(%arg0: tensor<1x36x384x64xf32>) -> tensor<1x12x384x64xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.slice"(%arg0) {axes = [1], ends = [12], starts = [0]} : (tensor<1x36x384x64xf32>) -> tensor<1x12x384x64xf32>
    return %0 : tensor<1x12x384x64xf32>
  }

  // CHECK-LABEL: func.func @func_slice2
  // CHECK: tosa.slice
  func.func @func_slice2(%arg0: tensor<1x36x384x64xf32>) -> tensor<1x12x100x64xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.slice"(%arg0) {axes = [1, 2], ends = [12, 284], starts = [0, 184]} : (tensor<1x36x384x64xf32>) -> tensor<1x12x100x64xf32>
    return %0 : tensor<1x12x100x64xf32>
  }
}

// -----

// Unary operations

module {
  // CHECK-LABEL: func.func @func_abs
  // CHECK: tosa.abs
  func.func @func_abs(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.abs"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @func_ceil
  // CHECK: tosa.ceil
  func.func @func_ceil(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.ceil"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @func_convert
  // CHECK: tosa.cast
  func.func @func_convert(%arg0: tensor<16xf16>) -> tensor<16xf32> {
    %0 = "migraphx.convert"(%arg0) : (tensor<16xf16>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @func_div_f32
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  func.func @func_div_f32(%arg0: tensor<1x36x384x64xf32>, %arg1: tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.div"(%arg0, %arg1) : (tensor<1x36x384x64xf32>, tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32>
    return %0 : tensor<1x36x384x64xf32>
  }

  // CHECK-LABEL: func.func @func_div_f16
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  func.func @func_div_f16(%arg0: tensor<1x36x384x64xf16>, %arg1: tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16> attributes{kernel, arch = ""} {
    %0 = "migraphx.div"(%arg0, %arg1) : (tensor<1x36x384x64xf16>, tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16>
    return %0 : tensor<1x36x384x64xf16>
  }

  // CHECK-LABEL: func.func @func_div_i32
  // CHECK: tosa.div
  func.func @func_div_i32(%arg0: tensor<1x36x384x64xi32>, %arg1: tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32> attributes{kernel, arch = ""} {
    %0 = "migraphx.div"(%arg0, %arg1) : (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32>
    return %0 : tensor<1x36x384x64xi32>
  }

  // CHECK-LABEL: func.func @func_erf_f32
  // CHECK: tosa.erf
  func.func @func_erf_f32(%arg0: tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.erf"(%arg0) : (tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32>
    return %0 : tensor<1x36x384x64xf32>
  }

  // CHECK-LABEL: func.func @func_erf_f16
  // CHECK: tosa.erf
  func.func @func_erf_f16(%arg0: tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16> attributes{kernel, arch = ""} {
    %0 = "migraphx.erf"(%arg0) : (tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16>
    return %0 : tensor<1x36x384x64xf16>
  }

  // CHECK-LABEL: func.func @func_exp_f32
  // CHECK: tosa.exp
  func.func @func_exp_f32(%arg0: tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.exp"(%arg0) : (tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32>
    return %0 : tensor<1x36x384x64xf32>
  }

  // CHECK-LABEL: func.func @func_exp_f16
  // CHECK: tosa.exp
  func.func @func_exp_f16(%arg0: tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16> attributes{kernel, arch = ""} {
    %0 = "migraphx.exp"(%arg0) : (tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16>
    return %0 : tensor<1x36x384x64xf16>
  }

  // CHECK-LABEL: func.func @func_floor
  // CHECK: tosa.floor
  func.func @func_floor(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.floor"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @func_log_f32
  // CHECK: tosa.log
  func.func @func_log_f32(%arg0: tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.log"(%arg0) : (tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32>
    return %0 : tensor<1x36x384x64xf32>
  }

  // CHECK-LABEL: func.func @func_log_f16
  // CHECK: tosa.log
  func.func @func_log_f16(%arg0: tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16> attributes{kernel, arch = ""} {
    %0 = "migraphx.log"(%arg0) : (tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16>
    return %0 : tensor<1x36x384x64xf16>
  }

  // CHECK-LABEL: func.func @func_neg_f32
  // CHECK: tosa.negate
  func.func @func_neg_f32(%arg0: tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32> attributes{kernel, arch = ""} {
    %0 = "migraphx.neg"(%arg0) : (tensor<1x36x384x64xf32>) -> tensor<1x36x384x64xf32>
    return %0 : tensor<1x36x384x64xf32>
  }

  // CHECK-LABEL: func.func @func_neg_f16
  // CHECK: tosa.negate
  func.func @func_neg_f16(%arg0: tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16> attributes{kernel, arch = ""} {
    %0 = "migraphx.neg"(%arg0) : (tensor<1x36x384x64xf16>) -> tensor<1x36x384x64xf16>
    return %0 : tensor<1x36x384x64xf16>
  }

  // CHECK-LABEL: func.func @func_power
  // CHECK: tosa.pow
  func.func @func_power(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.pow"(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @func_recip
  // CHECK: tosa.recip
  func.func @func_recip(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.recip"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @func_rsqrt
  // CHECK: tosa.rsqrt
  func.func @func_rsqrt(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.rsqrt"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @func_sigmoid
  // CHECK: tosa.sigmoid
  func.func @func_sigmoid(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.sigmoid"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }


  // CHECK-LABEL: func.func @func_rsqrt_opt
  // CHECK: tosa.rsqrt
  // CHECK-NOT: tosa.reciprocal
  func.func @func_rsqrt_opt(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.sqrt"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
    %1 = "migraphx.recip"(%0) : (tensor<16xf32>) -> tensor<16xf32>
     return %1 : tensor<16xf32>
  }
}

// -----

// Less trivial pointwise ops
module {
  // CHECK-LABEL: func.func @func_softmax_1d
  // CHECK-DAG: [[REDUCE_MAX:%[a-z0-9]+]] = "tosa.reduce_max"([[INPUT:%[a-z0-9]+]])
  // CHECK-DAG: [[SUB:%[a-z0-9]+]] = "tosa.sub"([[INPUT]], [[REDUCE_MAX]])
  // CHECK-DAG: [[EXP:%[a-z0-9]+]] = "tosa.exp"([[SUB]])
  // CHECK-DAG: [[REDUCE_SUM:%[a-z0-9]+]] = "tosa.reduce_sum"([[EXP]])
  // CHECK-DAG: [[RECIPROCAL:%[a-z0-9]+]] = "tosa.reciprocal"([[REDUCE_SUM]])
  // CHECK-DAG: "tosa.mul"([[EXP]], [[RECIPROCAL]])
  func.func @func_softmax_1d(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.softmax"(%arg0) {axis = 0 : i64} : (tensor<16xf32>) -> tensor<16xf32>
     return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func.func @func_softmax_4d
  // CHECK-DAG: [[REDUCE_MAX:%[a-z0-9]+]] = "tosa.reduce_max"([[INPUT:%[a-z0-9]+]])
  // CHECK-DAG: [[SUB:%[a-z0-9]+]] = "tosa.sub"([[INPUT]], [[REDUCE_MAX]])
  // CHECK-DAG: [[EXP:%[a-z0-9]+]] = "tosa.exp"([[SUB]])
  // CHECK-DAG: [[REDUCE_SUM:%[a-z0-9]+]] = "tosa.reduce_sum"([[EXP]])
  // CHECK-DAG: [[RECIPROCAL:%[a-z0-9]+]] = "tosa.reciprocal"([[REDUCE_SUM]])
  // CHECK-DAG: "tosa.mul"([[EXP]], [[RECIPROCAL]])
  func.func @func_softmax_4d(%arg0: tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32> {
    %0 = "migraphx.softmax"(%arg0) {axis = 1 : i64} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
     return %0 : tensor<16x16x16x16xf32>
  }
}
