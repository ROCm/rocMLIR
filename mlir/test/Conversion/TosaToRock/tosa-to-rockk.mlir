// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file --tosa-to-rock --rock-view-to-transform -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: mlir_bwd_conv2d
// CHECK: %[[buf:.*]] = bufferization.alloc_tensor() : tensor<1x32x32x512xf32>
// CHECK: %[[bufTransformed:.*]] = rock.transform %[[buf]] by #{{.*}} : tensor<1x32x32x512xf32> to tensor<1x32x32x1x512xf32>    
// CHECK: %[[convRes:.*]] = rock.conv_bwd_data(%{{.*}}, %[[bufTransformed]], %{{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], kernelId = 0 : index, output_layout = ["no", "ho", "wo", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index], usesV4R1 = false} : tensor<1x512x4x4x512xf32>, tensor<1x32x32x1x512xf32>, tensor<1x16x16x1x512xf32> -> tensor<1x32x32x1x512xf32>    
// CHECK: %[[transposeRes:.*]] = tosa.transpose %[[buf]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x512xf32>) -> tensor<1x512x32x32xf32>
// CHECK: %[[reshapeRes:.*]] = tosa.reshape %[[transposeRes]]
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
