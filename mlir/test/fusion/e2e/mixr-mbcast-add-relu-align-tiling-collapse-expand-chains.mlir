// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise --rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s
// ALLOW_RETRIES: 2

module {
    // CHECK: rock.threadwise_read_into 
    // CHECK: linalg.generic
    // CHECK: rock.threadwise_write_all
    // CHECK-NOT: memref.copy

    func.func @test(%arg0: tensor<1x4x1x1xf32>, %arg1: tensor<4x3x3x3xf32>, %arg2: tensor<4x3x3x3xf32>) -> tensor<4x4x1x1xf32> attributes {arch = "gfx908:sramecc+:xnack-", kernel = "mixr"} {
        %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : (tensor<1x4x1x1xf32>) -> tensor<4x4x1x1xf32>
        %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1], xdlopsV2 = true} : (tensor<4x3x3x3xf32>, tensor<4x3x3x3xf32>) -> tensor<4x4x1x1xf32>
        %2 = migraphx.add(%1, %0) : (tensor<4x4x1x1xf32>, tensor<4x4x1x1xf32>) -> tensor<4x4x1x1xf32>
        %3 = migraphx.relu(%2) : (tensor<4x4x1x1xf32>) -> tensor<4x4x1x1xf32>
        return %3 : tensor<4x4x1x1xf32>
    }
}
