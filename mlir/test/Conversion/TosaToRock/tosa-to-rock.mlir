// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file --tosa-to-rock --rock-view-to-transform -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_fusion
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x128x3x3x8xf32>, tensor<128x32x32x1x8xf32>, tensor<128x30x30x1x128xf32> -> tensor<128x30x30x1x128xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<128x30x30x1x128xf32> to tensor<128x30x30x128xf32>
// CHECK-NEXT: tosa.abs %[[castRes]]

func.func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>) -> tensor<128x30x30x128xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %zero = arith.constant dense<0.0> : tensor<128xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<128x30x30x128xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>
  %2 = "tosa.abs"(%1) {} : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>

  return %2 : tensor<128x30x30x128xf32>
}

// -----

// CHECK-LABEL: mlir_conv3d
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) {dilations = [1 : index, 1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "2", "c"], input_layout = ["ni", "0i", "1i", "2i", "gi", "ci"], output_layout = ["no", "0o", "1o", "2o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index, 1 : index]} : tensor<1x4x2x2x2x3xf32>, tensor<2x5x5x5x1x3xf32>, tensor<2x2x2x2x1x4xf32> -> tensor<2x2x2x2x1x4xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<2x2x2x2x1x4xf32> to tensor<2x2x2x2x4xf32>

func.func private @mlir_conv3d(%arg0: tensor<4x1x1x1x1xf32>, %arg1: tensor<2x5x5x5x3xf32>, %arg2: tensor<4x2x2x2x3xf32>) -> tensor<2x2x2x2x4xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %8 = tosa.conv3d %arg1, %arg2, %7, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1, 1>, group = 1 : i64, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x5x5x5x3xf32>, tensor<4x2x2x2x3xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x2x2x2x4xf32>
  return %8 : tensor<2x2x2x2x4xf32>
}

// CHECK-LABEL: mlir_conv1d
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [3 : index, 3 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x64x7x1x3xf32>, tensor<1x224x1x1x3xf32>, tensor<1x224x1x1x64xf32> -> tensor<1x224x1x1x64xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<1x224x1x1x64xf32> to tensor<1x224x1x64xf32>
// CHECK-NEXT: %[[reshapeRes:.*]] = tosa.reshape %[[castRes]], %{{.*}} : (tensor<1x224x1x64xf32>, !tosa.shape<3>) -> tensor<1x224x64xf32>

func.func private @mlir_conv1d(%arg0: tensor<64xf32>, %arg1: tensor<672xf32>, %arg2: tensor<1344xf32>) -> tensor<14336xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
    %const_shape = tosa.const_shape {values = dense<[64, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %const_shape : (tensor<64xf32>, !tosa.shape<3>) -> tensor<64x1x1xf32> 
    %2 = tosa.transpose %0 {perms = array<i32: 2, 0, 1>} : (tensor<64x1x1xf32>) -> tensor<1x64x1xf32>
    %3 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x64x224xf32>}> : () -> tensor<1x64x224xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = tosa.mul %3, %2, %shift : (tensor<1x64x224xf32>, tensor<1x64x1xf32>, tensor<1xi8>) -> tensor<1x64x224xf32>
    %const_shape2 = tosa.const_shape {values = dense<[64, 3, 7]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.reshape %arg2, %const_shape2 : (tensor<1344xf32>, !tosa.shape<3>) -> tensor<64x3x7xf32> 
    %const_shape3 = tosa.const_shape {values = dense<[1, 3, 224]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.reshape %arg1, %const_shape3 : (tensor<672xf32>, !tosa.shape<3>) -> tensor<1x3x224xf32> 
    %8 = tosa.transpose %6 {perms = array<i32: 0, 2, 1>} : (tensor<1x3x224xf32>) -> tensor<1x224x3xf32>
    %9 = tosa.transpose %5 {perms = array<i32: 0, 2, 1>} : (tensor<64x3x7xf32>) -> tensor<64x7x3xf32>
    %const_shape4 = tosa.const_shape {values = dense<[1, 224, 1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.reshape %8, %const_shape4 : (tensor<1x224x3xf32>, !tosa.shape<4>) -> tensor<1x224x1x3xf32> 
    %const_shape5 = tosa.const_shape {values = dense<[64, 7, 1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %11 = tosa.reshape %9, %const_shape5 : (tensor<64x7x3xf32>, !tosa.shape<4>) -> tensor<64x7x1x3xf32> 
    %12 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %13 = tosa.conv2d %10, %11, %12, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 3, 3, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x224x1x3xf32>, tensor<64x7x1x3xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x224x1x64xf32>
    %const_shape6 = tosa.const_shape {values = dense<[1, 224, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %14 = tosa.reshape %13, %const_shape6 : (tensor<1x224x1x64xf32>, !tosa.shape<3>) -> tensor<1x224x64xf32> 
    %15 = tosa.transpose %14 {perms = array<i32: 0, 2, 1>} : (tensor<1x224x64xf32>) -> tensor<1x64x224xf32>
    %16 = tosa.add %15, %4 : (tensor<1x64x224xf32>, tensor<1x64x224xf32>) -> tensor<1x64x224xf32>
    %const_shape7 = tosa.const_shape {values = dense<[14336]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %17 = tosa.reshape %16, %const_shape7 : (tensor<1x64x224xf32>, !tosa.shape<1>) -> tensor<14336xf32> 
    return %17 : tensor<14336xf32>
}

// -----

// CHECK-LABEL: mlir_bwd_conv1d
// CHECK: %[[buf:.*]] = bufferization.alloc_tensor() : tensor<1x224x1x64xf32>
// CHECK: %[[bufTransformed:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x224x1x64xf32> to tensor<1x224x1x1x64xf32>
// CHECK: %[[convRes:.*]] = rock.conv_bwd_data(%{{.*}}, %[[bufTransformed]], %{{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], kernelId = 0 : index, output_layout = ["no", "ho", "wo", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index], usesV4R1 = false} : tensor<1x64x1x1x3xf32>, tensor<1x224x1x1x64xf32>, tensor<1x224x1x1x3xf32> -> tensor<1x224x1x1x64xf32>
// CHECK-NEXT: %[[reshapeRes:.*]] = tosa.reshape %[[buf]]
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[reshapeRes]] by #{{.*}} : tensor<1x224x64xf32> to tensor<1x64x224xf32>

func.func @mlir_bwd_conv1d(%arg0: tensor<64xf32>, %arg1: tensor<672xf32>, %arg2: tensor<192xf32>) -> tensor<14336xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<14336> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.const_shape  {values = dense<[1, 224, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %4 = tosa.const_shape  {values = dense<[64, 1, 1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %5 = tosa.const_shape  {values = dense<[1, 224, 1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %6 = tosa.const_shape  {values = dense<[1, 3, 224]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x64x224xf32>}> : () -> tensor<1x64x224xf32>
  %8 = tosa.const_shape  {values = dense<[1, 64, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %9 = tosa.reshape %arg0, %8 : (tensor<64xf32>, !tosa.shape<3>) -> tensor<1x64x1xf32>
  %10 = tosa.add %9, %7 : (tensor<1x64x1xf32>, tensor<1x64x224xf32>) -> tensor<1x64x224xf32>
  %11 = tosa.reshape %arg1, %6 : (tensor<672xf32>, !tosa.shape<3>) -> tensor<1x3x224xf32>
  %12 = tosa.transpose %11 {perms = array<i32: 0, 2, 1>} : (tensor<1x3x224xf32>) -> tensor<1x224x3xf32>
  %13 = tosa.reshape %12, %5 : (tensor<1x224x3xf32>, !tosa.shape<4>) -> tensor<1x224x1x3xf32>
  %14 = tosa.reshape %arg2, %4 : (tensor<192xf32>, !tosa.shape<4>) -> tensor<64x1x1x3xf32>
  %15 = tosa.custom %13, %14, %2, %3, %3 {acc_type = f32, conv_kind = "bwd_data", dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x224x1x3xf32>, tensor<64x1x1x3xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x224x1x64xf32>
  %16 = tosa.reshape %15, %1 : (tensor<1x224x1x64xf32>, !tosa.shape<3>) -> tensor<1x224x64xf32>
  %17 = tosa.transpose %16 {perms = array<i32: 0, 2, 1>} : (tensor<1x224x64xf32>) -> tensor<1x64x224xf32>
  %18 = tosa.add %17, %10 : (tensor<1x64x224xf32>, tensor<1x64x224xf32>) -> tensor<1x64x224xf32>
  %19 = tosa.reshape %18, %0 : (tensor<1x64x224xf32>, !tosa.shape<1>) -> tensor<14336xf32>
  return %19 : tensor<14336xf32>
}

// -----

// CHECK-LABEL: mlir_bwd_conv2d
// CHECK: %[[buf:.*]] = bufferization.alloc_tensor() : tensor<1x32x32x512xf32>
// CHECK: %[[bufTransformed:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x32x32x1x512xf32>    
// CHECK: %[[convRes:.*]] = rock.conv_bwd_data(%{{.*}}, %[[bufTransformed]], %{{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], kernelId = 0 : index, output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index], usesV4R1 = false} : tensor<1x512x4x4x512xf32>, tensor<1x32x32x1x512xf32>, tensor<1x16x16x1x512xf32> -> tensor<1x32x32x1x512xf32>    
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x512x32x32xf32>
// CHECK-NEXT: %[[reshapeRes:.*]] = tosa.reshape %[[transRes]]
func.func @mlir_bwd_conv2d(%arg0: tensor<131072xf32>, %arg1: tensor<4194304xf32>) -> tensor<524288xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<[512, 512, 4, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.reshape %arg1, %0 : (tensor<4194304xf32>, !tosa.shape<4>) -> tensor<512x512x4x4xf32>
  %2 = tosa.const_shape  {values = dense<[1, 512, 16, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %3 = tosa.reshape %arg0, %2 : (tensor<131072xf32>, !tosa.shape<4>) -> tensor<1x512x16x16xf32>
  %4 = tosa.transpose %3 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x16x16xf32>) -> tensor<1x16x16x512xf32>
  %5 = tosa.transpose %1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<512x512x4x4xf32>) -> tensor<512x4x4x512xf32>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %8 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
  %9 = tosa.custom %4, %5, %8, %6, %7 {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x16x16x512xf32>, tensor<512x4x4x512xf32>, tensor<512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x512xf32>
  %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x512xf32>) -> tensor<1x512x32x32xf32>
  %11 = tosa.const_shape  {values = dense<524288> : tensor<1xindex>} : () -> !tosa.shape<1>
  %12 = tosa.reshape %10, %11 : (tensor<1x512x32x32xf32>, !tosa.shape<1>) -> tensor<524288xf32>
  return %12 : tensor<524288xf32>
}

// -----

// CHECK-LABEL: mlir_bwd_conv2d_stride
// CHECK: %[[buf:.*]] = bufferization.alloc_tensor() : tensor<1x32x32x512xf32>
// CHECK: %[[bufTransformed:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x32x32x1x512xf32>
// CHECK: %[[convRes:.*]] = rock.conv_bwd_data(%{{.*}}, %[[bufTransformed]], %{{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], kernelId = 0 : index, output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [2 : index, 2 : index], usesV4R1 = false} : tensor<1x512x4x4x512xf32>, tensor<1x32x32x1x512xf32>, tensor<1x16x16x1x512xf32> -> tensor<1x32x32x1x512xf32>
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x512x32x32xf32>
// CHECK-NEXT: %[[reshapeRes:.*]] = tosa.reshape %[[transRes]]
func.func @mlir_bwd_conv2d_stride(%arg0: tensor<131072xf32>, %arg1: tensor<4194304xf32>) -> tensor<524288xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<[512, 512, 4, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.reshape %arg1, %0 : (tensor<4194304xf32>, !tosa.shape<4>) -> tensor<512x512x4x4xf32>
  %2 = tosa.const_shape  {values = dense<[1, 512, 16, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %3 = tosa.reshape %arg0, %2 : (tensor<131072xf32>, !tosa.shape<4>) -> tensor<1x512x16x16xf32>
  %4 = tosa.transpose %3 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x16x16xf32>) -> tensor<1x16x16x512xf32>
  %5 = tosa.transpose %1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<512x512x4x4xf32>) -> tensor<512x4x4x512xf32>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %8 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
  %9 = tosa.custom %4, %5, %8, %6, %7 {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x16x16x512xf32>, tensor<512x4x4x512xf32>, tensor<512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x512xf32>
  %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x512xf32>) -> tensor<1x512x32x32xf32>
  %11 = tosa.const_shape  {values = dense<524288> : tensor<1xindex>} : () -> !tosa.shape<1>
  %12 = tosa.reshape %10, %11 : (tensor<1x512x32x32xf32>, !tosa.shape<1>) -> tensor<524288xf32>
  return %12 : tensor<524288xf32>
}

// -----
// CHECK-LABEL: mlir_bwd_conv2d_group
// CHECK: %[[buf:.*]] = bufferization.alloc_tensor() : tensor<1x32x32x512xf32>
// CHECK: %[[bufTransformed:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x32x32x2x256xf32>
// CHECK: %[[convRes:.*]] = rock.conv_bwd_data(%{{.*}}, %[[bufTransformed]], %{{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], kernelId = 0 : index, output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index], usesV4R1 = false} : tensor<2x256x4x4x512xf32>, tensor<1x32x32x2x256xf32>, tensor<1x16x16x2x256xf32> -> tensor<1x32x32x2x256xf32>
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x512x32x32xf32>
// CHECK-NEXT: %[[reshapeRes:.*]] = tosa.reshape %[[transRes]]
func.func @mlir_bwd_conv2d_group(%arg0: tensor<131072xf32>, %arg1: tensor<4194304xf32>) -> tensor<524288xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<[512, 512, 4, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.reshape %arg1, %0 : (tensor<4194304xf32>, !tosa.shape<4>) -> tensor<512x512x4x4xf32>
  %2 = tosa.const_shape  {values = dense<[1, 512, 16, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %3 = tosa.reshape %arg0, %2 : (tensor<131072xf32>, !tosa.shape<4>) -> tensor<1x512x16x16xf32>
  %4 = tosa.transpose %3 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x16x16xf32>) -> tensor<1x16x16x512xf32>
  %5 = tosa.transpose %1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<512x512x4x4xf32>) -> tensor<512x4x4x512xf32>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %8 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
  %9 = tosa.custom %4, %5, %8, %6, %7 {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 2 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x16x16x512xf32>, tensor<512x4x4x512xf32>, tensor<512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x512xf32>
  %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x512xf32>) -> tensor<1x512x32x32xf32>
  %11 = tosa.const_shape  {values = dense<524288> : tensor<1xindex>} : () -> !tosa.shape<1>
  %12 = tosa.reshape %10, %11 : (tensor<1x512x32x32xf32>, !tosa.shape<1>) -> tensor<524288xf32>
  return %12 : tensor<524288xf32>
}

// -----
// CHECK-LABEL: mlir_bwd_conv2d_dilation
// CHECK: %[[buf:.*]] = bufferization.alloc_tensor() : tensor<1x32x32x512xf32>
// CHECK: %[[bufTransformed:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x32x32x1x512xf32>
// CHECK: %[[convRes:.*]] = rock.conv_bwd_data(%{{.*}}, %[[bufTransformed]], %{{.*}}) {dilations = [2 : index, 2 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], kernelId = 0 : index, output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index], usesV4R1 = false} : tensor<1x512x4x4x512xf32>, tensor<1x32x32x1x512xf32>, tensor<1x16x16x1x512xf32> -> tensor<1x32x32x1x512xf32>
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x512x32x32xf32>
// CHECK-NEXT: %[[reshapeRes:.*]] = tosa.reshape %[[transRes]]
func.func @mlir_bwd_conv2d_dilation(%arg0: tensor<131072xf32>, %arg1: tensor<4194304xf32>) -> tensor<524288xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<[512, 512, 4, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.reshape %arg1, %0 : (tensor<4194304xf32>, !tosa.shape<4>) -> tensor<512x512x4x4xf32>
  %2 = tosa.const_shape  {values = dense<[1, 512, 16, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %3 = tosa.reshape %arg0, %2 : (tensor<131072xf32>, !tosa.shape<4>) -> tensor<1x512x16x16xf32>
  %4 = tosa.transpose %3 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x16x16xf32>) -> tensor<1x16x16x512xf32>
  %5 = tosa.transpose %1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<512x512x4x4xf32>) -> tensor<512x4x4x512xf32>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %8 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
  %9 = tosa.custom %4, %5, %8, %6, %7 {acc_type = f32, dilation = array<i64: 2, 2>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x16x16x512xf32>, tensor<512x4x4x512xf32>, tensor<512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x512xf32>
  %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x512xf32>) -> tensor<1x512x32x32xf32>
  %11 = tosa.const_shape  {values = dense<524288> : tensor<1xindex>} : () -> !tosa.shape<1>
  %12 = tosa.reshape %10, %11 : (tensor<1x512x32x32xf32>, !tosa.shape<1>) -> tensor<524288xf32>
  return %12 : tensor<524288xf32>
}

// -----

// CHECK-LABEL: mlir_bwd_conv3d
// CHECK: %[[buf:.*]] = bufferization.alloc_tensor() : tensor<1x4x4x4x16xf32>
// CHECK: %[[bufTransformed:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x4x4x4x16xf32> to tensor<1x4x4x4x1x16xf32>
// CHECK: %[[convRes:.*]] = rock.conv_bwd_data(%{{.*}}, %[[bufTransformed]], %{{.*}}) {dilations = [1 : index, 1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "2", "c"], input_layout = ["ni", "0i", "1i", "2i", "gi", "ci"], kernelId = 0 : index, output_layout = ["no", "0o", "1o", "2o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index, 1 : index], usesV4R1 = false} : tensor<1x1x4x4x4x16xf32>, tensor<1x4x4x4x1x16xf32>, tensor<16x1x1x1x1x16xf32> -> tensor<1x4x4x4x1x16xf32>
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x4x4x4x16xf32> to tensor<1x16x4x4x4xf32>
// CHECK-NEXT: %[[reshapeRes:.*]] = tosa.reshape %[[transRes]]
func.func @mlir_bwd_conv3d(%arg0: tensor<1024xf32>, %arg1: tensor<256xf32>) -> tensor<1024xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<[16, 1, 1, 1, 16]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %1 = tosa.const_shape  {values = dense<1024> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<16xf32>}> : () -> tensor<16xf32>
  %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %4 = tosa.const_shape  {values = dense<[1, 16, 4, 4, 4]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %5 = tosa.reshape %arg0, %4 : (tensor<1024xf32>, !tosa.shape<5>) -> tensor<1x16x4x4x4xf32>
  %6 = tosa.reshape %arg1, %0 : (tensor<256xf32>, !tosa.shape<5>) -> tensor<16x1x1x1x16xf32>
  %7 = tosa.transpose %5 {perms = array<i32: 0, 2, 3, 4, 1>} : (tensor<1x16x4x4x4xf32>) -> tensor<1x4x4x4x16xf32>
  %8 = tosa.custom %6, %7, %2, %3, %3 {acc_type = f32, conv_kind = "bwd_data", dilation = array<i64: 1, 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<16x1x1x1x16xf32>, tensor<1x4x4x4x16xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x4x16xf32>
  %9 = tosa.transpose %8 {perms = array<i32: 0, 4, 1, 2, 3>} : (tensor<1x4x4x4x16xf32>) -> tensor<1x16x4x4x4xf32>
  %10 = tosa.reshape %9, %1 : (tensor<1x16x4x4x4xf32>, !tosa.shape<1>) -> tensor<1024xf32>
  return %10 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: mlir_dot_transpose_add
// CHECK: %[[gemmRes:.*]] = rock.gemm %{{.*}} = %{{.*}} * %{{.*}} storeMethod =  set : tensor<1x4x5xf32> = tensor<1x4x5xf32> * tensor<1x5x5xf32> -> tensor<1x4x5xf32>
// CHECK-NEXT: %{{.*}} = tosa.reshape %[[gemmRes]]
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[gemmRes]] by #{{.*}} : tensor<1x4x5xf32> to tensor<1x5x4xf32>

func.func private @mlir_dot_transpose_add(%arg0: tensor<20xf32>, %arg1: tensor<20xf32>, %arg2: tensor<25xf32>) -> (tensor<20xf32>, tensor<20xf32>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %const_shape = "tosa.const_shape"() {values = dense<[1, 5, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %const_shape) : (tensor<20xf32>, !tosa.shape<3>) -> tensor<1x5x4xf32> 
  %const_shape2 = "tosa.const_shape"() {values = dense<[1, 5, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = "tosa.reshape"(%arg2, %const_shape2) : (tensor<25xf32>, !tosa.shape<3>) -> tensor<1x5x5xf32> 
  %const_shape3 = "tosa.const_shape"() {values = dense<[1, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = "tosa.reshape"(%arg1, %const_shape3) : (tensor<20xf32>, !tosa.shape<3>) -> tensor<1x4x5xf32> 
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %3 = "tosa.matmul"(%2, %1, %a_zp, %b_zp) : (tensor<1x4x5xf32>, tensor<1x5x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x5xf32>
  %const_shape4 = "tosa.const_shape"() {values = dense<[20]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %4 = "tosa.reshape"(%3, %const_shape4) : (tensor<1x4x5xf32>, !tosa.shape<1>) -> tensor<20xf32> 
  %6 = "tosa.transpose"(%3) {perms = array<i32: 0, 2, 1>} : (tensor<1x4x5xf32>) -> tensor<1x5x4xf32>
  %7 = "tosa.add"(%6, %0) : (tensor<1x5x4xf32>, tensor<1x5x4xf32>) -> tensor<1x5x4xf32>
  %8 = "tosa.reshape"(%7, %const_shape4) : (tensor<1x5x4xf32>, !tosa.shape<1>) -> tensor<20xf32> 
  return %4, %8 : tensor<20xf32>, tensor<20xf32>
}

// -----

// CHECK-LABEL: mlir_conv_transpose_add
// CHECK: %[[convRes:.*]] = rock.conv(%{{.*}}, %{{.*}}, %{{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x128x3x3x8xf32>, tensor<128x32x32x1x8xf32>, tensor<128x30x30x1x128xf32> -> tensor<128x30x30x1x128xf32>
// CHECK-NEXT: %[[castRes:.*]] = rock.tensor_untransform_cast %[[convRes]] aka %{{.*}} : tensor<128x30x30x1x128xf32> to tensor<128x30x30x128xf32>
// CHECK-NEXT: %{{.*}} = tosa.reshape %[[castRes]]
// CHECK-NEXT: %[[transRes:.*]] = rock.transform %[[castRes]] by #{{.*}} : tensor<128x30x30x128xf32> to tensor<128x30x128x30xf32>

func.func @mlir_conv_transpose_add(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>, %arg2: tensor<128x30x128x30xf32>) -> (tensor<14745600xf32>, tensor<14745600xf32>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %zero = arith.constant dense<0.0> : tensor<128xf32>

  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %zero, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<128x30x30x128xf32>
  %const_shape = "tosa.const_shape"() {values = dense<[14745600]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = "tosa.reshape"(%0, %const_shape) : (tensor<128x30x30x128xf32>, !tosa.shape<1>) -> tensor<14745600xf32> 
  %6 = "tosa.transpose"(%0) {perms = array<i32: 0, 1, 3, 2>} : (tensor<128x30x30x128xf32>) -> tensor<128x30x128x30xf32>
  %7 = "tosa.add"(%6, %arg2) : (tensor<128x30x128x30xf32>, tensor<128x30x128x30xf32>) -> tensor<128x30x128x30xf32>
  %8 = "tosa.reshape"(%7, %const_shape) : (tensor<128x30x128x30xf32>, !tosa.shape<1>) -> tensor<14745600xf32> 

  return %1, %8 : tensor<14745600xf32>, tensor<14745600xf32>
}

// -----

// CHECK-LABEL: mlir_scaled_gemm_both_scales
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by %{{.*}}

func.func @mlir_scaled_gemm_both_scales(%arg0: tensor<1x128x256xf4E2M1FN>, %arg1: tensor<1x256x512xf4E2M1FN>, 
                                        %scaleA: tensor<1x128x256xf8E8M0FNU>, %scaleB: tensor<1x256x512xf8E8M0FNU>) 
                                        -> tensor<1x128x512xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  // Cast fp4 to f32
  %0 = tosa.cast %arg0 : (tensor<1x128x256xf4E2M1FN>) -> tensor<1x128x256xf32>
  %1 = tosa.cast %arg1 : (tensor<1x256x512xf4E2M1FN>) -> tensor<1x256x512xf32>
  
  // Cast scales to f32
  %2 = tosa.cast %scaleA : (tensor<1x128x256xf8E8M0FNU>) -> tensor<1x128x256xf32>
  %3 = tosa.cast %scaleB : (tensor<1x256x512xf8E8M0FNU>) -> tensor<1x256x512xf32>
  
  // Multiply by scales
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %4 = tosa.mul %0, %2, %shift : (tensor<1x128x256xf32>, tensor<1x128x256xf32>, tensor<1xi8>) -> tensor<1x128x256xf32>
  %5 = tosa.mul %1, %3, %shift : (tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1xi8>) -> tensor<1x256x512xf32>
  
  // MatMul
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %6 = tosa.matmul %4, %5, %a_zp, %b_zp : (tensor<1x128x256xf32>, tensor<1x256x512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x128x512xf32>
  
  return %6 : tensor<1x128x512xf32>
}

// -----

// CHECK-LABEL: mlir_scaled_gemm_batched
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by %{{.*}}

func.func @mlir_scaled_gemm_batched(%arg0: tensor<4x128x256xf4E2M1FN>, %arg1: tensor<4x256x512xf4E2M1FN>, 
                                    %scaleA: tensor<4x128x256xf8E8M0FNU>, %scaleB: tensor<4x256x512xf8E8M0FNU>) 
                                    -> tensor<4x128x512xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  // Cast fp4 to f32
  %0 = tosa.cast %arg0 : (tensor<4x128x256xf4E2M1FN>) -> tensor<4x128x256xf32>
  %1 = tosa.cast %arg1 : (tensor<4x256x512xf4E2M1FN>) -> tensor<4x256x512xf32>
  
  // Cast scales to f32
  %2 = tosa.cast %scaleA : (tensor<4x128x256xf8E8M0FNU>) -> tensor<4x128x256xf32>
  %3 = tosa.cast %scaleB : (tensor<4x256x512xf8E8M0FNU>) -> tensor<4x256x512xf32>
  
  // Multiply by scales
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %4 = tosa.mul %0, %2, %shift : (tensor<4x128x256xf32>, tensor<4x128x256xf32>, tensor<1xi8>) -> tensor<4x128x256xf32>
  %5 = tosa.mul %1, %3, %shift : (tensor<4x256x512xf32>, tensor<4x256x512xf32>, tensor<1xi8>) -> tensor<4x256x512xf32>
  
  // Batched MatMul
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %6 = tosa.matmul %4, %5, %a_zp, %b_zp : (tensor<4x128x256xf32>, tensor<4x256x512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x128x512xf32>
  
  return %6 : tensor<4x128x512xf32>
}

// -----

// CHECK-LABEL: mlir_scaled_gemm_with_transpose_a
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by %{{.*}}

func.func @mlir_scaled_gemm_with_transpose_a(%arg0: tensor<1x256x128xf4E2M1FN>, %arg1: tensor<1x256x512xf4E2M1FN>, 
                                              %scaleA: tensor<1x256x128xf8E8M0FNU>, %scaleB: tensor<1x256x512xf8E8M0FNU>) 
                                              -> tensor<1x128x512xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  // Transpose A from [1, 256, 128] to [1, 128, 256]
  %transposed_a = tosa.transpose %arg0 {perms = array<i32: 0, 2, 1>} : (tensor<1x256x128xf4E2M1FN>) -> tensor<1x128x256xf4E2M1FN>
  %transposed_scale_a = tosa.transpose %scaleA {perms = array<i32: 0, 2, 1>} : (tensor<1x256x128xf8E8M0FNU>) -> tensor<1x128x256xf8E8M0FNU>
  
  // Cast fp4 to f32
  %0 = tosa.cast %transposed_a : (tensor<1x128x256xf4E2M1FN>) -> tensor<1x128x256xf32>
  %1 = tosa.cast %arg1 : (tensor<1x256x512xf4E2M1FN>) -> tensor<1x256x512xf32>
  
  // Cast scales to f32
  %2 = tosa.cast %transposed_scale_a : (tensor<1x128x256xf8E8M0FNU>) -> tensor<1x128x256xf32>
  %3 = tosa.cast %scaleB : (tensor<1x256x512xf8E8M0FNU>) -> tensor<1x256x512xf32>
  
  // Multiply by scales
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %4 = tosa.mul %0, %2, %shift : (tensor<1x128x256xf32>, tensor<1x128x256xf32>, tensor<1xi8>) -> tensor<1x128x256xf32>
  %5 = tosa.mul %1, %3, %shift : (tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1xi8>) -> tensor<1x256x512xf32>
  
  // MatMul
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %6 = tosa.matmul %4, %5, %a_zp, %b_zp : (tensor<1x128x256xf32>, tensor<1x256x512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x128x512xf32>
  
  return %6 : tensor<1x128x512xf32>
}

// -----

// CHECK-LABEL: mlir_scaled_gemm_with_transpose_b
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by %{{.*}}

func.func @mlir_scaled_gemm_with_transpose_b(%arg0: tensor<1x128x256xf4E2M1FN>, %arg1: tensor<1x512x256xf4E2M1FN>, 
                                              %scaleA: tensor<1x128x256xf8E8M0FNU>, %scaleB: tensor<1x512x256xf8E8M0FNU>) 
                                              -> tensor<1x128x512xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  // Transpose B from [1, 512, 256] to [1, 256, 512]
  %transposed_b = tosa.transpose %arg1 {perms = array<i32: 0, 2, 1>} : (tensor<1x512x256xf4E2M1FN>) -> tensor<1x256x512xf4E2M1FN>
  %transposed_scale_b = tosa.transpose %scaleB {perms = array<i32: 0, 2, 1>} : (tensor<1x512x256xf8E8M0FNU>) -> tensor<1x256x512xf8E8M0FNU>
  
  // Cast fp4 to f32
  %0 = tosa.cast %arg0 : (tensor<1x128x256xf4E2M1FN>) -> tensor<1x128x256xf32>
  %1 = tosa.cast %transposed_b : (tensor<1x256x512xf4E2M1FN>) -> tensor<1x256x512xf32>
  
  // Cast scales to f32
  %2 = tosa.cast %scaleA : (tensor<1x128x256xf8E8M0FNU>) -> tensor<1x128x256xf32>
  %3 = tosa.cast %transposed_scale_b : (tensor<1x256x512xf8E8M0FNU>) -> tensor<1x256x512xf32>
  
  // Multiply by scales
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %4 = tosa.mul %0, %2, %shift : (tensor<1x128x256xf32>, tensor<1x128x256xf32>, tensor<1xi8>) -> tensor<1x128x256xf32>
  %5 = tosa.mul %1, %3, %shift : (tensor<1x256x512xf32>, tensor<1x256x512xf32>, tensor<1xi8>) -> tensor<1x256x512xf32>
  
  // MatMul
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %6 = tosa.matmul %4, %5, %a_zp, %b_zp : (tensor<1x128x256xf32>, tensor<1x256x512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x128x512xf32>
  
  return %6 : tensor<1x128x512xf32>
}

// -----

// CHECK-LABEL: mlir_scaled_gemm_with_a_broadcast
// CHECK-DAG: %[[A_BROADCAST:.*]] = rock.transform %{{.*}} by #{{.*}} : tensor<1x1x512xf4E2M1FN> to tensor<1x16x512xf4E2M1FN>
// CHECK-DAG: %[[A_SCALE_BROADCAST:.*]] = rock.transform %{{.*}} by #{{.*}} : tensor<1x1x512xf8E8M0FNU> to tensor<1x16x512xf8E8M0FNU>
// CHECK: rock.gemm %{{.*}} = %[[A_BROADCAST]] scaled by %[[A_SCALE_BROADCAST]] * %{{.*}} scaled by %{{.*}}

func.func @mlir_scaled_gemm_with_a_broadcast(%arg0: tensor<8192xf4E2M1FN>, %arg1: tensor<8192xf4E2M1FN>, %arg2: tensor<8192xf8E8M0FNU>, %arg3: tensor<8192xf8E8M0FNU>) -> tensor<256xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<256> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.const_shape  {values = dense<[1, 1, 512]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
  %5 = tosa.const_shape  {values = dense<[1, 16, 512]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %6 = tosa.const_shape  {values = dense<[1, 512, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [1, 512, 16] : tensor<8192xf8E8M0FNU> into tensor<1x512x16xf8E8M0FNU>
  %expanded_0 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 16, 512] : tensor<8192xf8E8M0FNU> into tensor<1x16x512xf8E8M0FNU>
  %extracted_slice = tensor.extract_slice %expanded_0[0, 0, 0] [1, 1, 512] [1, 1, 1] : tensor<1x16x512xf8E8M0FNU> to tensor<1x1x512xf8E8M0FNU>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 512, 16] : tensor<8192xf4E2M1FN> into tensor<1x512x16xf4E2M1FN>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 16, 512] : tensor<8192xf4E2M1FN> into tensor<1x16x512xf4E2M1FN>
  %extracted_slice_3 = tensor.extract_slice %expanded_2[0, 0, 0] [1, 1, 512] [1, 1, 1] : tensor<1x16x512xf4E2M1FN> to tensor<1x1x512xf4E2M1FN>
  %7 = tosa.cast %extracted_slice_3 : (tensor<1x1x512xf4E2M1FN>) -> tensor<1x1x512xf32>
  %8 = tosa.cast %expanded_1 : (tensor<1x512x16xf4E2M1FN>) -> tensor<1x512x16xf32>
  %9 = tosa.cast %extracted_slice : (tensor<1x1x512xf8E8M0FNU>) -> tensor<1x1x512xf32>
  %10 = tosa.mul %7, %9, %2 : (tensor<1x1x512xf32>, tensor<1x1x512xf32>, tensor<1xi8>) -> tensor<1x1x512xf32>
  %11 = tosa.cast %expanded : (tensor<1x512x16xf8E8M0FNU>) -> tensor<1x512x16xf32>
  %12 = tosa.mul %8, %11, %2 : (tensor<1x512x16xf32>, tensor<1x512x16xf32>, tensor<1xi8>) -> tensor<1x512x16xf32>
  %13 = tosa.matmul %10, %12, %1, %1 {acc_type = f32} : (tensor<1x1x512xf32>, tensor<1x512x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x16x16xf32>
  %collapsed = tensor.collapse_shape %13 [[0, 1, 2]] : tensor<1x16x16xf32> into tensor<256xf32>
  return %collapsed : tensor<256xf32>
}

// -----

// CHECK-LABEL: mlir_quant_dot_fp4_fp32_scales
// CHECK-DAG: rock.transform %{{.*}} by #{{.*}} : tensor<196608xf4E2M1FN> to tensor<1x256x768xf4E2M1FN>
// CHECK-DAG: rock.transform %{{.*}} by #{{.*}} : tensor<38608896xf4E2M1FN> to tensor<1x768x50272xf4E2M1FN>
// CHECK-DAG: rock.transform %{{.*}} by #{{.*}} : tensor<6144xf32> to tensor<1x8x1x768xf32>
// CHECK-DAG: rock.transform %{{.*}} by #{{.*}} : tensor<1206528xf32> to tensor<1x768x1571x1xf32>
// CHECK-DAG: %[[A_SCALE:.*]] = tosa.cast %{{.*}} : (tensor<1x256x768xf32>) -> tensor<1x256x768xf8E8M0FNU>
// CHECK-DAG: %[[B_SCALE:.*]] = tosa.cast %{{.*}} : (tensor<1x768x50272xf32>) -> tensor<1x768x50272xf8E8M0FNU>
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by %{{.*}} storeMethod =  set
func.func @mlir_quant_dot_fp4_fp32_scales(%arg0: tensor<196608xf4E2M1FN>, %arg1: tensor<38608896xf4E2M1FN>, %arg2: tensor<6144xf32>, %arg3: tensor<1206528xf32>) -> tensor<12869632xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %0 = tosa.const_shape  {values = dense<12869632> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %2 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x768x1571x32xf32>}> : () -> tensor<1x768x1571x32xf32>
  %3 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x8x32x768xf32>}> : () -> tensor<1x8x32x768xf32>
  %4 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %5 = tosa.const_shape  {values = dense<[1, 768, 50272]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %6 = tosa.const_shape  {values = dense<[1, 8, 1, 768]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %7 = tosa.const_shape  {values = dense<[1, 768, 1571, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %8 = tosa.const_shape  {values = dense<[1, 256, 768]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 256, 768] : tensor<196608xf4E2M1FN> into tensor<1x256x768xf4E2M1FN>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 768, 1571, 1] : tensor<1206528xf32> into tensor<1x768x1571x1xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 8, 1, 768] : tensor<6144xf32> into tensor<1x8x1x768xf32>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 768, 50272] : tensor<38608896xf4E2M1FN> into tensor<1x768x50272xf4E2M1FN>
  %9 = tosa.mul %expanded_1, %3, %4 : (tensor<1x8x1x768xf32>, tensor<1x8x32x768xf32>, tensor<1xi8>) -> tensor<1x8x32x768xf32>
  %collapsed = tensor.collapse_shape %9 [[0], [1, 2], [3]] : tensor<1x8x32x768xf32> into tensor<1x256x768xf32>
  %10 = tosa.mul %expanded_0, %2, %4 : (tensor<1x768x1571x1xf32>, tensor<1x768x1571x32xf32>, tensor<1xi8>) -> tensor<1x768x1571x32xf32>
  %collapsed_3 = tensor.collapse_shape %10 [[0], [1], [2, 3]] : tensor<1x768x1571x32xf32> into tensor<1x768x50272xf32>
  %11 = tosa.cast %collapsed : (tensor<1x256x768xf32>) -> tensor<1x256x768xf8E8M0FNU>
  %12 = tosa.cast %collapsed_3 : (tensor<1x768x50272xf32>) -> tensor<1x768x50272xf8E8M0FNU>
  %13 = tosa.cast %expanded : (tensor<1x256x768xf4E2M1FN>) -> tensor<1x256x768xf32>
  %14 = tosa.cast %11 : (tensor<1x256x768xf8E8M0FNU>) -> tensor<1x256x768xf32>
  %15 = tosa.mul %13, %14, %4 : (tensor<1x256x768xf32>, tensor<1x256x768xf32>, tensor<1xi8>) -> tensor<1x256x768xf32>
  %16 = tosa.cast %expanded_2 : (tensor<1x768x50272xf4E2M1FN>) -> tensor<1x768x50272xf32>
  %17 = tosa.cast %12 : (tensor<1x768x50272xf8E8M0FNU>) -> tensor<1x768x50272xf32>
  %18 = tosa.mul %16, %17, %4 : (tensor<1x768x50272xf32>, tensor<1x768x50272xf32>, tensor<1xi8>) -> tensor<1x768x50272xf32>
  %19 = tosa.matmul %15, %18, %1, %1 {acc_type = f32} : (tensor<1x256x768xf32>, tensor<1x768x50272xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x50272xf32>
  %collapsed_4 = tensor.collapse_shape %19 [[0, 1, 2]] : tensor<1x256x50272xf32> into tensor<12869632xf32>
  return %collapsed_4 : tensor<12869632xf32>
}

