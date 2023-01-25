// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-opt --rock-fold-transpose -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK-DAG: #[[MAP1:.*]] = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 1, 28, 28] -> [1, 1, 1, 1]>
  // CHECK: rock.transforming_for {{.*}} (%[[loadCoord:.*]]) = {{.*}}#[[MAP1]]
  // CHECK-SAME: bounds [1, 1, 1, 1]
  // CHECK-SAME: strides [1, 1, 1, 1]
  // CHECK-NEXT: arith.andi
  // CHECK-NEXT: rock.global_load %arg0[%[[loadCoord]]]
  // CHECK: linalg.generic{{.*}} outs(%[[outBuf:.*]] : memref<1xf32, 5>)
  // CHECK: global_store %[[outBuf]]{{.*}} -> %arg3
  func.func @main(%arg0: tensor<1x1x1x1xf32>, %arg1: tensor<1x3x32x32xf32>, %arg2: tensor<1x3x5x5xf32>) -> tensor<1x1x28x28xf32> attributes {arch = "gfx908:sramecc+:xnack-", kernel = "mixr"} {
    %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [1, 1, 28, 28]} : (tensor<1x1x1x1xf32>) -> tensor<1x1x28x28xf32>
    %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1], xdlopsV2 = true} : (tensor<1x3x32x32xf32>, tensor<1x3x5x5xf32>) -> tensor<1x1x28x28xf32>
    %2 = migraphx.add(%1, %0) : (tensor<1x1x28x28xf32>, tensor<1x1x28x28xf32>) -> tensor<1x1x28x28xf32>
    %3 = migraphx.relu(%2) : (tensor<1x1x28x28xf32>) -> tensor<1x1x28x28xf32>
    return %3 : tensor<1x1x28x28xf32>
  }
}
