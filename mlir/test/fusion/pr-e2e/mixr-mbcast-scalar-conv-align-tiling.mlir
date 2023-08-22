// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK-COUNT-4: rock.threadwise_read_into {{.*}}
  // CHECK: rock.threadwise_read_into 
  // CHECK: linalg.generic
  // CHECK: rock.threadwise_write_all
  // CHECK-NOT: memref.copy

  func.func @main(%arg0: !migraphx.shaped<1x1x1x1xf32, 1x1x1x1>, %arg1: !migraphx.shaped<1x3x32x32xf32, 3072x1024x32x1>, %arg2: !migraphx.shaped<1x3x5x5xf32, 75x25x5x1>) -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1> attributes {arch = "gfx908:sramecc+:xnack-", kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [1, 1, 28, 28]} : !migraphx.shaped<1x1x1x1xf32, 1x1x1x1> -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
    %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1], xdlopsV2 = true} : !migraphx.shaped<1x3x32x32xf32, 3072x1024x32x1>, !migraphx.shaped<1x3x5x5xf32, 75x25x5x1> -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
    %2 = migraphx.add %1, %0 : !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>, !migraphx.shaped<1x1x28x28xf32, 784x784x28x1> -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
    %3 = migraphx.relu %2 : !migraphx.shaped<1x1x28x28xf32, 784x784x28x1> -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
    return %3 : !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
  }
}
