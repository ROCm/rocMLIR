// RUN: rocmlir-opt %s | FileCheck %s

module {
  func.func @main(%arg0: !migraphx.shaped<1x3x224x224xf32, 150528x50176x224x1>) -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1> {
    %0 = migraphx.literal (dense<0.0> : tensor<64x1xf32>) : <64x3x7x7xf32, 147x49x7x1>
    %1 = migraphx.literal (dense<0.0> : tensor<64x1xf32>) : <64x1xf32, 1x1>
    %2 = migraphx.literal (dense<0.0> : tensor<64x1xf32>) : <64x1xf32, 1x1>
    %3 = migraphx.literal (dense<0.0> : tensor<64x1xf32>) : <64x1xf32, 1x1>
    %4 = migraphx.literal (dense<0.0> : tensor<64x1xf32>) : <64x1xf32, 1x1>
    %5 = migraphx.convolution %arg0, %0 {padding = [3:i64, 3:i64], stride = [2:i64, 2:i64], dilation = [1:i64, 1:i64], group = 1:i64} : <1x3x224x224xf32, 150528x50176x224x1>, <64x3x7x7xf32, 147x49x7x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    %6 = migraphx.batch_norm_inference %5, %1, %2, %3, %4 {epsilon = 0.00001:f32, momentum = 0.9:f32, bn_mode = 1:i64} : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>, !migraphx.shaped<64x1xf32, 1x1>, !migraphx.shaped<64x1xf32, 1x1>, !migraphx.shaped<64x1xf32, 1x1>, !migraphx.shaped<64x1xf32, 1x1>-> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
    %7 = migraphx.relu %6 : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    %8 = migraphx.pooling %7 {mode = "max", padding = [1:i64, 1:i64], stride = [2:i64, 2:i64], length = [3:i64, 3:i64], ceil_mode = 0:i64}: <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    return %8 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
  }
}
// CHECK-LABEL: func @main
// CHECK: migraphx.literal
// CHECK-NEXT: migraphx.literal
// CHECK-NEXT: migraphx.literal
// CHECK-NEXT: migraphx.literal
// CHECK-NEXT: migraphx.literal
// CHECK-NEXT: migraphx.convolution
// CHECK-NEXT: migraphx.batch_norm_inference
// CHECK-NEXT: migraphx.relu
// CHECK-NEXT: migraphx.pooling
