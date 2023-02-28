// RUN: rocmlir-opt --tosa-to-rock %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_fusion
// CHECK: %[[convRes:.*]] = rock.conv2d(%{{.*}}, %{{.*}}, %{{.*}}) features = {{none|xdlops}} {arch = {{.*}}, dilations = [1 : i32, 1 : i32], filter_layout = ["k", "y", "x", "c", "g"], input_layout = ["ni", "hi", "wi", "ci", "gi"], output_layout = ["no", "ho", "wo", "ko", "go"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : tensor<128x8x3x3x1xf32>, tensor<128x8x32x32x1xf32>, tensor<128x128x30x30x1xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<128x128x30x30x1xf32> to tensor<128x128x30x30xf32>
// CHECK-NEXT: "tosa.abs"(%[[castRes]])

func.func @test_fusion(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>) -> tensor<128x128x30x30xf32> attributes {kernel, arch = ""} {
  %zero = arith.constant dense<0.0> : tensor<128xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<128xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  %2 = "tosa.abs"(%1) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>

  return %2 : tensor<128x128x30x30xf32>
}

// -----
