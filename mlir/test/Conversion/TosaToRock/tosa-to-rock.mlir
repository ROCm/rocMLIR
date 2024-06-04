// RUN: rocmlir-opt -split-input-file --tosa-to-rock %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_fusion
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features = {{none|xdlops}} {arch = {{.*}}, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x128x8x3x3xf32>, tensor<128x8x32x1x32xf32>, tensor<128x128x30x1x30xf32> -> tensor<128x128x30x1x30xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<128x128x30x1x30xf32> to tensor<128x128x30x30xf32>
// CHECK-NEXT: tosa.abs %[[castRes]]

func.func @test_fusion(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>) -> tensor<128x128x30x30xf32> attributes {kernel, arch = ""} {
  %zero = arith.constant dense<0.0> : tensor<128xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<128xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  %2 = "tosa.abs"(%1) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>

  return %2 : tensor<128x128x30x30xf32>
}

// -----

// CHECK-LABEL: mlir_conv3d
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features =  none {arch = "", dilations = [1 : index, 1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "2", "c"], input_layout = ["ni", "0i", "1i", "2i", "gi", "ci"], output_layout = ["no", "0o", "1o", "2o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index, 1 : index]} : tensor<1x4x2x2x2x3xf32>, tensor<2x5x5x5x1x3xf32>, tensor<2x2x2x2x1x4xf32> -> tensor<2x2x2x2x1x4xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<2x2x2x2x1x4xf32> to tensor<2x2x2x2x4xf32>

func.func private @mlir_conv3d(%arg0: tensor<4x1x1x1x1xf32>, %arg1: tensor<2x5x5x5x3xf32>, %arg2: tensor<4x2x2x2x3xf32>) -> tensor<2x2x2x2x4xf32> attributes {kernel, arch = ""} {
  %7 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
  %8 = tosa.conv3d %arg1, %arg2, %7 {dilation = array<i64: 1, 1, 1>, group = 1 : i64, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x5x5x5x3xf32>, tensor<4x2x2x2x3xf32>, tensor<4xf32>) -> tensor<2x2x2x2x4xf32>
  return %8 : tensor<2x2x2x2x4xf32>
}
