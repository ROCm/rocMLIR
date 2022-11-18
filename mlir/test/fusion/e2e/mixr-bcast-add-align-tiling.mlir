// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-opt --rock-fold-transpose -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s

module {
    // CHECK-COUNT-2: %{{.*}} = rock.alloc() : memref<4xf32, 5>
    // CHECK-NEXT: %[[vecBuf:.*]] = rock.alloc() : memref<4xf32, 5>
    // CHECK-NEXT: rock.transforming_for {forceUnroll, useIndexDiffs}
    // CHECK-NEXT: %[[ldVal:.*]] = rock.global_load
    // CHECK-NEXT: rock.in_bounds_store %[[ldVal]] -> %[[vecBuf]]
    func.func @test(%arg0: tensor<64xf32>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> attributes{kernel, arch = ""} {
        %0 = migraphx.broadcast(%arg0) {axis = 1:i64, out_lens= [1:i64, 64:i64, 112:i64, 112:i64] } : (tensor<64xf32>)-> tensor<1x64x112x112xf32>
        %1 = migraphx.convolution(%arg1, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
        %2 = migraphx.add(%1, %0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
        %3 = migraphx.relu(%2) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
        return %3 : tensor<1x64x112x112xf32>
    }
}
