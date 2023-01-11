// RUN: rocmlir-opt --migraphx-transform --canonicalize --migraphx-to-tosa %s -verify-diagnostics -o -| FileCheck %s

module  {
  // CHECK-LABEL: func.func @matmul
  // CHECK: tosa.matmul
  func.func @matmul(%arg0: tensor<2x256x384xf32>, %arg1: tensor<2x384x768xf32>) -> tensor<2x256x768xf32> {
    %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<2x256x384xf32>, tensor<2x384x768xf32>) -> tensor<2x256x768xf32>
     return %0 : tensor<2x256x768xf32>
  }

  // CHECK-LABEL: func.func @matmul_larger_batch
  // CHECK: tosa.matmul
  func.func @matmul_larger_batch(%arg0: tensor<2x16x256x384xf32>, %arg1: tensor<2x16x384x768xf32>) -> tensor<2x16x256x768xf32> {
    %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<2x16x256x384xf32>, tensor<2x16x384x768xf32>) -> tensor<2x16x256x768xf32>
     return %0 : tensor<2x16x256x768xf32>
  }

  // CHECK-LABEL: func.func @matmul_rank2
  // CHECK: tosa.matmul
  func.func @matmul_rank2(%arg0: tensor<32x72xf32>, %arg1: tensor<72x64xf32>) -> tensor<32x64xf32> {
    %0 = "migraphx.dot"(%arg0, %arg1) : (tensor<32x72xf32>, tensor<72x64xf32>) -> tensor<32x64xf32>
     return %0 : tensor<32x64xf32>
  }

  // CHECK-LABEL: func.func @matmul_broadcast
  func.func @matmul_broadcast(%arg0: tensor<64x64x2304xf16>, %arg1: tensor<64x64x768xf16>, %arg2: tensor<1x768x2304xf16>) -> tensor<64x64x2304xf16> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    %0 = migraphx.multibroadcast(%arg2) {out_dyn_dims = [], out_lens = [64, 768, 2304]} : (tensor<1x768x2304xf16>) -> tensor<64x768x2304xf16>
    // CHECK-DAG: %[[RESHAPE0:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 4096, 768]}
    // CHECK-DAG: %[[RESHAPE1:.*]] = "tosa.reshape"(%arg2) {new_shape = [1, 768, 2304]}
    %1 = migraphx.dot(%arg1, %0) : tensor<64x64x768xf16>, tensor<64x768x2304xf16> -> tensor<64x64x2304xf16>
    // CHECK-DAG: %[[MATMUL:.*]] = "tosa.matmul"(%[[RESHAPE0]], %[[RESHAPE1]]
    // CHECK: %[[RESHAPE2:.*]] = "tosa.reshape"(%[[MATMUL]]) {new_shape = [64, 64, 2304]}
    %2 = migraphx.add(%1, %arg0) : (tensor<64x64x2304xf16>, tensor<64x64x2304xf16>) -> tensor<64x64x2304xf16>
    return %2 : tensor<64x64x2304xf16>
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

  // CHECK-LABEL: func.func @func_sqrt
  // CHECK: tosa.rsqrt
  // CHECK-NOT: tosa.reciprocal
  func.func @func_sqrt(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "migraphx.sqrt"(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
    %1 = "migraphx.recip"(%0) : (tensor<16xf32>) -> tensor<16xf32>
     return %1 : tensor<16xf32>
  }

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

  // broadcast ops will be lowered as implicit broadcast in tosa, passes if they're converted and legalize tosa.
  // CHECK-LABEL: func @func_mbcast
  func.func @func_mbcast(%arg0: tensor<1x64x1x1xf32>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> attributes {kernel = "mixr"} {
    %0 = migraphx.multibroadcast(%arg0) {out_lens = [1, 64, 112, 112]} : (tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %2 = migraphx.add(%1, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %3 = migraphx.relu(%2) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %3 : tensor<1x64x112x112xf32>
  }
}

// -----

