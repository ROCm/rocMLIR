// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise --rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s

module {
    // CHECK-DAG: #[[MAP1:.*]] = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 64, 112, 112] -> [1, 64, 1, 1]>
    // CHECK: rock.transforming_for {{.*}} (%[[loadCoord:.*]]) = {{.*}}#[[MAP1]]
    // CHECK-SAME: bounds [1, 1, 1, 4]
    // CHECK-SAME: strides [1, 1, 1, 1]
    // CHECK-NEXT: rock.global_load %arg0[%[[loadCoord]]]
    // CHECK: linalg.generic{{.*}} outs(%[[outBuf:.*]] : memref<4xf32, 5>)
    // CHECK: global_store %[[outBuf]]{{.*}} -> %arg3
    func.func @test(%arg0: tensor<1x64x1x1xf32>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> attributes{kernel, arch = ""} {
        %0 = migraphx.multibroadcast(%arg0) {out_lens = [1, 64, 112, 112]} : (tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
        %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
        %2 = migraphx.add(%1, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
        %3 = migraphx.relu(%2) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
        return %3 : tensor<1x64x112x112xf32>
    }
}
