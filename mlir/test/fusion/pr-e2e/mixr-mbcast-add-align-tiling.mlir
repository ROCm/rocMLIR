// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise --rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s
// ALLOW_RETRIES: 2

module {
    // CHECK-COUNT-4: rock.threadwise_read_into {{.*}}
    // CHECK: rock.threadwise_read_into
    // CHECK: linalg.generic
    // CHECK: rock.threadwise_write_all
    // CHECK-NOT: memref.copy
    func.func @test(%arg0: !migraphx.shaped<1x64x1x1xf32, 64x1x1x1>, %arg1: !migraphx.shaped<1x3x224x224xf32, 150528x50176x224x1>, %arg2: !migraphx.shaped<64x3x7x7xf32, 147x49x7x1>) -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1> attributes{kernel, arch = ""} {
        %0 = migraphx.multibroadcast %arg0 {out_lens = [1, 64, 112, 112]} : !migraphx.shaped<1x64x1x1xf32, 64x1x1x1> -> !migraphx.shaped<1x64x112x112xf32, 0x1x0x0>
        %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : !migraphx.shaped<1x3x224x224xf32, 150528x50176x224x1>, !migraphx.shaped<64x3x7x7xf32, 147x49x7x1> -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
        %2 = migraphx.add %1, %0 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>, !migraphx.shaped<1x64x112x112xf32, 0x1x0x0> -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
        %3 = migraphx.relu %2 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1> -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
        return %3 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
    }
}
