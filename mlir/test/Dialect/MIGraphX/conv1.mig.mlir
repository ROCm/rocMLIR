module {
  func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x56x56xf32> {
    %0 = "migraphx.constant"(){shape = [64:i32, 3:i32, 7:i32, 7:i32], type = f32}: () -> tensor<64x3x7x7xf32>
    %1 = "migraphx.constant"(){shape = [64:i32, 1:i32], type = f32}: () -> tensor<64x1xf32>
    %2 = "migraphx.constant"(){shape = [64:i32, 1:i32], type = f32}: () -> tensor<64x1xf32>
    %3 = "migraphx.constant"(){shape = [64:i32, 1:i32], type = f32}: () -> tensor<64x1xf32>
    %4 = "migraphx.constant"(){shape = [64:i32, 1:i32], type = f32}: () -> tensor<64x1xf32>
    %5 = "migraphx.convolution"(%arg0, %0) {padding = [3:i32, 3:i32], stride = [2:i32, 2:i32], dilation = [1:i32, 1:i32], group = 1:i32, padding_mode = 0:i32} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %6 = "migraphx.batch_norm_inference"(%5, %1, %2, %3, %4) {epsilon = 0.00001:f32, momentum = 0.9:f32, bn_mode = 1:i32} : (tensor<1x64x112x112xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>)-> tensor<1x64x112x112xf32>
    %7 = "migraphx.relu"(%6) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %8 = "migraphx.pooling"(%7) {mode = "max", padding = [1:i32, 1:i32], stride = [2:i32, 2:i32], lengths = [3:i32, 3:i32], ceil_mode = 0:i32}: (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %8 : tensor<1x64x112x112xf32>
  }
}