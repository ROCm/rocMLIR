// RUN: mlir-opt --split-input-file --tosa-to-miopen-on-tensors %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_fusion
// CHECK: miopen.conv2d(%{{.*}}, %{{.*}}, %{{.*}}) {arch = {{.*}}, dilations = [1 : i32, 1 : i32], filter_layout = ["k", "y", "x", "c", "g"], input_layout = ["ni", "hi", "wi", "ci", "gi"], num_cu = {{.*}} : i32, output_layout = ["no", "ho", "wo", "ko", "go"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3x1xf32>, memref<128x8x32x32x1xf32>, memref<128x128x30x30x1xf32>
    
func @test_fusion(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  %2 = "tosa.abs"(%1) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>

  return %2 : tensor<128x128x30x30xf32>
}

// -----

