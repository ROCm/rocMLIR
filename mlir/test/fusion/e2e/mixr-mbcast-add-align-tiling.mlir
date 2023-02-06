// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise --rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s
// ALLOW_RETRIES: 2

module {
    // CHECK: rock.threadwise_read_into 
    // CHECK: linalg.generic
    // CHECK: rock.threadwise_write_all
    // CHECK-NOT: memref.copy
    func.func @test(%arg0: tensor<1x64x1x1xf32>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> attributes{kernel, arch = ""} {
        %0 = migraphx.multibroadcast(%arg0) {out_lens = [1, 64, 112, 112]} : (tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
        %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
        %2 = migraphx.add(%1, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
        %3 = migraphx.relu(%2) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
        return %3 : tensor<1x64x112x112xf32>
    }
}
