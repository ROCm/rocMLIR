// RUN: rocmlir-opt -split-input-file --tosa-to-rock --rock-view-to-transform %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_fusion
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features = {{none|xdlops}} {arch = {{.*}}, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x128x8x3x3xf32>, tensor<128x8x32x1x32xf32>, tensor<128x128x30x1x30xf32> -> tensor<128x128x30x1x30xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<128x128x30x1x30xf32> to tensor<128x128x30x30xf32>
// CHECK-NEXT: tosa.abs %[[castRes]]

func.func @test_fusion(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>) -> tensor<128x128x30x30xf32> attributes {kernel, arch = ""} {
  %zero = arith.constant dense<0.0> : tensor<128xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<128xf32>) -> tensor<128x128x30x30xf32>
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
  %8 = tosa.conv3d %arg1, %arg2, %7 {acc_type = f32, dilation = array<i64: 1, 1, 1>, group = 1 : i64, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x5x5x5x3xf32>, tensor<4x2x2x2x3xf32>, tensor<4xf32>) -> tensor<2x2x2x2x4xf32>
  return %8 : tensor<2x2x2x2x4xf32>
}

// CHECK-LABEL: mlir_conv1d
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features =  none {arch = "", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [3 : index, 3 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x64x7x1x3xf32>, tensor<1x224x1x1x3xf32>, tensor<1x224x1x1x64xf32> -> tensor<1x224x1x1x64xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<1x224x1x1x64xf32> to tensor<1x224x1x64xf32>
// CHECK-NEXT: %[[reshapeRes:.*]] = tosa.reshape %[[castRes]] {new_shape = array<i64: 1, 224, 64>} : (tensor<1x224x1x64xf32>) -> tensor<1x224x64xf32>

func.func private @mlir_conv1d(%arg0: tensor<64xf32>, %arg1: tensor<672xf32>, %arg2: tensor<1344xf32>) -> tensor<14336xf32> attributes {kernel, arch = ""} {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 64, 1, 1>} : (tensor<64xf32>) -> tensor<64x1x1xf32>
    %1 = "tosa.const"() <{value = dense<[2, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2 = tosa.transpose %0, %1 : (tensor<64x1x1xf32>, tensor<3xi32>) -> tensor<1x64x1xf32>
    %3 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x224xf32>}> : () -> tensor<1x64x224xf32>
    %4 = tosa.add %3, %2 : (tensor<1x64x224xf32>, tensor<1x64x1xf32>) -> tensor<1x64x224xf32>
    %5 = tosa.reshape %arg2 {new_shape = array<i64: 64, 3, 7>} : (tensor<1344xf32>) -> tensor<64x3x7xf32>
    %6 = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 224>} : (tensor<672xf32>) -> tensor<1x3x224xf32>
    %7 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %8 = tosa.transpose %6, %7 : (tensor<1x3x224xf32>, tensor<3xi32>) -> tensor<1x224x3xf32>
    %9 = tosa.transpose %5, %7 : (tensor<64x3x7xf32>, tensor<3xi32>) -> tensor<64x7x3xf32>
    %10 = tosa.reshape %8 {new_shape = array<i64: 1, 224, 1, 3>} : (tensor<1x224x3xf32>) -> tensor<1x224x1x3xf32>
    %11 = tosa.reshape %9 {new_shape = array<i64: 64, 7, 1, 3>} : (tensor<64x7x3xf32>) -> tensor<64x7x1x3xf32>
    %12 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %13 = tosa.conv2d %10, %11, %12 {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 3, 3, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x224x1x3xf32>, tensor<64x7x1x3xf32>, tensor<64xf32>) -> tensor<1x224x1x64xf32>
    %14 = tosa.reshape %13 {new_shape = array<i64: 1, 224, 64>} : (tensor<1x224x1x64xf32>) -> tensor<1x224x64xf32>
    %15 = tosa.transpose %14, %7 : (tensor<1x224x64xf32>, tensor<3xi32>) -> tensor<1x64x224xf32>
    %16 = tosa.add %15, %4 : (tensor<1x64x224xf32>, tensor<1x64x224xf32>) -> tensor<1x64x224xf32>
    %17 = tosa.reshape %16 {new_shape = array<i64: 14336>} : (tensor<1x64x224xf32>) -> tensor<14336xf32>
    return %17 : tensor<14336xf32>
}

// -----

// CHECK-LABEL: mlir_dot_transpose_add
// CHECK: %[[gemmRes:.*]] = rock.gemm %{{.*}} = %{{.*}} * %{{.*}} features =  none storeMethod =  set {arch = ""} : tensor<1x4x5xf32> = tensor<1x4x5xf32> * tensor<1x5x5xf32> -> tensor<1x4x5xf32>
// CHECK-NEXT: %{{.*}} = tosa.reshape %[[gemmRes]]
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[gemmRes]] by #{{.*}} : tensor<1x4x5xf32> to tensor<1x5x4xf32>

func.func private @mlir_dot_transpose_add(%arg0: tensor<20xf32>, %arg1: tensor<20xf32>, %arg2: tensor<25xf32>) -> (tensor<20xf32>, tensor<20xf32>) attributes {kernel, arch = ""} {
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 5, 4>} : (tensor<20xf32>) -> tensor<1x5x4xf32>
  %1 = "tosa.reshape"(%arg2) {new_shape = array<i64: 1, 5, 5>} : (tensor<25xf32>) -> tensor<1x5x5xf32>
  %2 = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 4, 5>} : (tensor<20xf32>) -> tensor<1x4x5xf32>
  %3 = "tosa.matmul"(%2, %1) : (tensor<1x4x5xf32>, tensor<1x5x5xf32>) -> tensor<1x4x5xf32>
  %4 = "tosa.reshape"(%3) {new_shape = array<i64: 20>} : (tensor<1x4x5xf32>) -> tensor<20xf32>
  %5 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
  %6 = "tosa.transpose"(%3, %5) : (tensor<1x4x5xf32>, tensor<3xi32>) -> tensor<1x5x4xf32>
  %7 = "tosa.add"(%6, %0) : (tensor<1x5x4xf32>, tensor<1x5x4xf32>) -> tensor<1x5x4xf32>
  %8 = "tosa.reshape"(%7) {new_shape = array<i64: 20>} : (tensor<1x5x4xf32>) -> tensor<20xf32>
  return %4, %8 : tensor<20xf32>, tensor<20xf32>
}

// -----

// CHECK-LABEL: mlir_conv_transpose_add
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) features = {{none|xdlops}} {arch = {{.*}}, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x128x8x3x3xf32>, tensor<128x8x32x1x32xf32>, tensor<128x128x30x1x30xf32> -> tensor<128x128x30x1x30xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<128x128x30x1x30xf32> to tensor<128x128x30x30xf32>
// CHECK-NEXT: %{{.*}} = tosa.reshape %[[castRes]]
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[castRes]] by #{{.*}} : tensor<128x128x30x30xf32> to tensor<128x30x128x30xf32>

func.func @mlir_conv_transpose_add(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<128x30x128x30xf32>) -> (tensor<14745600xf32>, tensor<14745600xf32>) attributes {kernel, arch = ""} {
  %zero = arith.constant dense<0.0> : tensor<128xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<128xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.reshape"(%0) {new_shape = array<i64: 14745600>} : (tensor<128x128x30x30xf32>) -> tensor<14745600xf32>
  %5 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %6 = "tosa.transpose"(%0, %5) : (tensor<128x128x30x30xf32>, tensor<4xi32>) -> tensor<128x30x128x30xf32>
  %7 = "tosa.add"(%6, %arg2) : (tensor<128x30x128x30xf32>, tensor<128x30x128x30xf32>) -> tensor<128x30x128x30xf32>
  %8 = "tosa.reshape"(%7) {new_shape = array<i64: 14745600>} : (tensor<128x30x128x30xf32>) -> tensor<14745600xf32>

  return %1, %8 : tensor<14745600xf32>, tensor<14745600xf32>
}
