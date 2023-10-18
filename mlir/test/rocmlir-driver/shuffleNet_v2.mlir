// The test was ectracted from the ShuffleNet_V2 model of MIGraphX. https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/issues/2315
// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-opt

module {
  func.func @mlir_reshape_transpose_reshape_convolution_add_relu(%arg0: tensor<1x116x1x1xf32>, %arg1: tensor<1x116x28x28xf32>, %arg2: tensor<116x116x1x1xf32>) -> tensor<1x116x28x28xf32> attributes {arch = "gfx1100", kernel = "mixr", num_cu = 48 : i64} {
    %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [1, 116, 28, 28]} : (tensor<1x116x1x1xf32>) -> tensor<1x116x28x28xf32>
    %1 = migraphx.reshape(%arg1) {dims = [1, 2, 58, 28, 28]} : (tensor<1x116x28x28xf32>) -> tensor<1x2x58x28x28xf32>
    %2 = migraphx.transpose(%1) {permutation = [0, 2, 1, 3, 4]} : (tensor<1x2x58x28x28xf32>) -> tensor<1x58x2x28x28xf32>
    %3 = migraphx.reshape(%2) {dims = [1, -1, 28, 28]} : (tensor<1x58x2x28x28xf32>) -> tensor<1x116x28x28xf32>
    %4 = migraphx.convolution(%3, %arg2) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x116x28x28xf32>, tensor<116x116x1x1xf32>) -> tensor<1x116x28x28xf32>
    %5 = migraphx.add(%4, %0) : (tensor<1x116x28x28xf32>, tensor<1x116x28x28xf32>) -> tensor<1x116x28x28xf32>
    %6 = migraphx.relu(%5) : (tensor<1x116x28x28xf32>) -> tensor<1x116x28x28xf32>
    return %6 : tensor<1x116x28x28xf32>
  }
}
