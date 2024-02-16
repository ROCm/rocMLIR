// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise --rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align --rock-pipeline | FileCheck %s
// ALLOW_RETRIES: 2

module {
    // CHECK-COUNT-4: rock.threadwise_read_into {{.*}}
    // CHECK: rock.threadwise_read_into
    // CHECK: linalg.generic
    // CHECK: rock.threadwise_write_all
    // CHECK-NOT: memref.copy

    func.func @test(%arg0: !migraphx.shaped<1x4x1x1xf32, 4x1x1x1>, %arg1: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>, %arg2: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>) -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1> attributes {arch = "gfx908:sramecc+:xnack-", kernel = "mixr"} {
        %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : <1x4x1x1xf32, 4x1x1x1> -> <4x4x1x1xf32, 0x1x1x1>
        %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1], xdlopsV2 = true} : <4x3x3x3xf32, 27x9x3x1>, <4x3x3x3xf32, 27x9x3x1> -> <4x4x1x1xf32, 4x1x1x1>
        %2 = migraphx.add %1, %0 : <4x4x1x1xf32, 4x1x1x1>, <4x4x1x1xf32, 0x1x1x1> -> <4x4x1x1xf32, 4x1x1x1>
        %3 = migraphx.relu %2 : <4x4x1x1xf32, 4x1x1x1> -> <4x4x1x1xf32, 4x1x1x1>
        return %3 : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
    }
}
