// RUN: rocmlir-opt --rocmlir-custom-tosa-decompose --split-input-file %s | FileCheck %s

// CHECK: @bwd_data_conv2d
// CHECK: %[[reverse1:.*]] = tosa.reverse %{{.*}} {axis = 1 : i32} : (tensor<2048x2x2x512xf32>) -> tensor<2048x2x2x512xf32>
// CHECK: %[[reverse2:.*]] = tosa.reverse %[[reverse1]] {axis = 2 : i32} : (tensor<2048x2x2x512xf32>) -> tensor<2048x2x2x512xf32>
// CHECK: tosa.conv2d %{{.*}}, %[[reverse2]], %{{.*}}, %{{.*}}, %{{.*}} {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x18x18x512xf32>, tensor<2048x2x2x512xf32>, tensor<2048xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x17x17x2048xf32>
func.func @bwd_data_conv2d(%arg0: tensor<131072xf32>, %arg1: tensor<4194304xf32>) -> tensor<524288xf32> {
  %0 = tosa.const_shape  {values = dense<[512, 512, 4, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.reshape %arg1, %0 : (tensor<4194304xf32>, !tosa.shape<4>) -> tensor<512x512x4x4xf32>
  %2 = tosa.const_shape  {values = dense<[1, 512, 16, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %3 = tosa.reshape %arg0, %2 : (tensor<131072xf32>, !tosa.shape<4>) -> tensor<1x512x16x16xf32>
  %4 = tosa.transpose %3 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x16x16xf32>) -> tensor<1x16x16x512xf32>
  %5 = tosa.transpose %1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<512x512x4x4xf32>) -> tensor<512x4x4x512xf32>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
  %8 = tosa.custom %4, %5, %7, %6, %6 {acc_type = f32, conv_kind = "bwd_data", dilation = array<i64: 1, 1>, domain_name = "rock", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x16x16x512xf32>, tensor<512x4x4x512xf32>, tensor<512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x512xf32>
  %9 = tosa.transpose %8 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x512xf32>) -> tensor<1x512x32x32xf32>
  %10 = tosa.const_shape  {values = dense<524288> : tensor<1xindex>} : () -> !tosa.shape<1>
  %11 = tosa.reshape %9, %10 : (tensor<1x512x32x32xf32>, !tosa.shape<1>) -> tensor<524288xf32>
  return %11 : tensor<524288xf32>
}

// -----
// CHECK: func @bwd_data_conv1d
// CHECK: %[[reverse1:.*]] = tosa.reverse %{{.*}} {axis = 1 : i32} : (tensor<64x1x1x3xf32>) -> tensor<64x1x1x3xf32>
// CHECK: %[[reverse2:.*]] = tosa.reverse %[[reverse1]] {axis = 2 : i32} : (tensor<64x1x1x3xf32>) -> tensor<64x1x1x3xf32>
// CHECK: tosa.conv2d %{{.*}}, %[[reverse2]], %{{.*}}, %{{.*}}, %{{.*}} {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x224x1x3xf32>, tensor<64x1x1x3xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x224x1x64xf32>
func.func @bwd_data_conv1d(%arg0: tensor<64xf32>, %arg1: tensor<672xf32>, %arg2: tensor<192xf32>) -> tensor<14336xf32> {
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
  %15 = tosa.custom %13, %14, %2, %3, %3 {acc_type = f32, conv_kind = "bwd_data", dilation = array<i64: 1, 1>, domain_name = "rock", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x224x1x3xf32>, tensor<64x1x1x3xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x224x1x64xf32>
  %16 = tosa.reshape %15, %1 : (tensor<1x224x1x64xf32>, !tosa.shape<3>) -> tensor<1x224x64xf32>
  %17 = tosa.transpose %16 {perms = array<i32: 0, 2, 1>} : (tensor<1x224x64xf32>) -> tensor<1x64x224xf32>
  %18 = tosa.add %17, %10 : (tensor<1x64x224xf32>, tensor<1x64x224xf32>) -> tensor<1x64x224xf32>
  %19 = tosa.reshape %18, %0 : (tensor<1x64x224xf32>, !tosa.shape<1>) -> tensor<14336xf32>
  return %19 : tensor<14336xf32>
}

