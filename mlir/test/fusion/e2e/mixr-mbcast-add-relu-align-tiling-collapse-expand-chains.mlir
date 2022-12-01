// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-opt --rock-fold-transpose -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s

module {
    // CHECK-DAG: #[[MAP:.*]] = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [4, 4, 1, 1] -> [1, 4, 1, 1]>
    // CHECK: rock.transforming_for{{.*}}#[[MAP]]
    // CHECK-NEXT: %[[ldVal:.*]] = rock.global_load{{.*}}: memref<1x4x1x1xf32> -> f32
    // CHECK-NEXT: rock.in_bounds_store %[[ldVal]]{{.*}}: f32 -> memref<1xf32, 5>, index
    func.func @test(%arg0: tensor<1x4x1x1xf32>, %arg1: tensor<4x3x3x3xf32>, %arg2: tensor<4x3x3x3xf32>) -> tensor<4x4x1x1xf32> attributes {arch = "gfx908:sramecc+:xnack-", kernel = "mixr"} {
        %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : (tensor<1x4x1x1xf32>) -> tensor<4x4x1x1xf32>
        %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1], xdlopsV2 = true} : (tensor<4x3x3x3xf32>, tensor<4x3x3x3xf32>) -> tensor<4x4x1x1xf32>
        %2 = migraphx.add(%1, %0) : (tensor<4x4x1x1xf32>, tensor<4x4x1x1xf32>) -> tensor<4x4x1x1xf32>
        %3 = migraphx.relu(%2) : (tensor<4x4x1x1xf32>) -> tensor<4x4x1x1xf32>
        return %3 : tensor<4x4x1x1xf32>
    }
}
