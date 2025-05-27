// RUN: rocmlir-opt --tosa-to-rock %s -verify-diagnostics -o -| FileCheck %s

// CHECK: rock.conv_elementwise_gemm
func.func @conv_gemm(%arg0: tensor<131072xf32>, %arg1: tensor<36864xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<64xf32>) -> tensor<32768xf32> attributes {kernel, arch = ""} {
  %0 = tosa.const_shape  {values = dense<[1, 16, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xf32> into tensor<1x16x64xf32>
  %1 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1>} : (tensor<1x16x64xf32>) -> tensor<1x64x16xf32>
  %2 = tosa.const_shape  {values = dense<[64, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
  %3 = tosa.transpose %expanded_0 {perms = array<i32: 2, 1, 0>} : (tensor<64x1x1xf32>) -> tensor<1x1x64xf32>
  %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2048x64xf32>}> : () -> tensor<1x2048x64xf32>
  %5 = tosa.add %3, %4 : (tensor<1x1x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %6 = tosa.const_shape  {values = dense<[64, 3, 3, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [64, 3, 3, 64] : tensor<36864xf32> into tensor<64x3x3x64xf32>
  %7 = tosa.transpose %expanded_1 {perms = array<i32: 0, 3, 1, 2>} : (tensor<64x3x3x64xf32>) -> tensor<64x64x3x3xf32>
  %8 = tosa.const_shape  {values = dense<[2, 32, 32, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 32, 32, 64] : tensor<131072xf32> into tensor<2x32x32x64xf32>
  %9 = tosa.transpose %expanded_2 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %10 = tosa.transpose %9 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %11 = tosa.transpose %7 {perms = array<i32: 0, 2, 3, 1>} : (tensor<64x64x3x3xf32>) -> tensor<64x3x3x64xf32>
  %12 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %13 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %14 = tosa.conv2d %10, %11, %13, %12, %12 {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<2x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x64xf32>
  %15 = tosa.transpose %14 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %16 = tosa.transpose %15 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %17 = tosa.const_shape  {values = dense<[1, 2048, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed = tensor.collapse_shape %16 [[0, 1, 2], [3]] : tensor<2x32x32x64xf32> into tensor<2048x64xf32>
  %expanded_3 = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [1, 2048, 64] : tensor<2048x64xf32> into tensor<1x2048x64xf32>
  %18 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %19 = tosa.mul %expanded_3, %5, %18 : (tensor<1x2048x64xf32>, tensor<1x2048x64xf32>, tensor<1xi8>) -> tensor<1x2048x64xf32>
  %20 = tosa.matmul %19, %1, %12, %12 : (tensor<1x2048x64xf32>, tensor<1x64x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x16xf32>
  %21 = tosa.const_shape  {values = dense<32768> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_4 = tensor.collapse_shape %20 [[0, 1, 2]] : tensor<1x2048x16xf32> into tensor<32768xf32>
  return %collapsed_4 : tensor<32768xf32>
}

// CHECK: rock.conv_elementwise_gemm
func.func @conv_gemm_no_scale(%arg0: tensor<131072xf32>, %arg1: tensor<36864xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<64xf32>) -> tensor<32768xf32> attributes {kernel, arch = ""} {
  %0 = tosa.const_shape  {values = dense<[1, 16, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xf32> into tensor<1x16x64xf32>
  %1 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1>} : (tensor<1x16x64xf32>) -> tensor<1x64x16xf32>
  %2 = tosa.const_shape  {values = dense<[64, 3, 3, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [64, 3, 3, 64] : tensor<36864xf32> into tensor<64x3x3x64xf32>
  %3 = tosa.transpose %expanded_0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<64x3x3x64xf32>) -> tensor<64x64x3x3xf32>
  %4 = tosa.const_shape  {values = dense<[2, 32, 32, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 32, 32, 64] : tensor<131072xf32> into tensor<2x32x32x64xf32>
  %5 = tosa.transpose %expanded_1 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %6 = tosa.transpose %5 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %7 = tosa.transpose %3 {perms = array<i32: 0, 2, 3, 1>} : (tensor<64x64x3x3xf32>) -> tensor<64x3x3x64xf32>
  %8 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %9 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %10 = tosa.conv2d %6, %7, %9, %8, %8 {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<2x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x64xf32>
  %11 = tosa.transpose %10 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %12 = tosa.transpose %11 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %13 = tosa.const_shape  {values = dense<[1, 2048, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed = tensor.collapse_shape %12 [[0, 1, 2], [3]] : tensor<2x32x32x64xf32> into tensor<2048x64xf32>
  %expanded_2 = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [1, 2048, 64] : tensor<2048x64xf32> into tensor<1x2048x64xf32>
  %14 = tosa.matmul %expanded_2, %1, %8, %8 : (tensor<1x2048x64xf32>, tensor<1x64x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x16xf32>
  %15 = tosa.const_shape  {values = dense<32768> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_3 = tensor.collapse_shape %14 [[0, 1, 2]] : tensor<1x2048x16xf32> into tensor<32768xf32>
  return %collapsed_3 : tensor<32768xf32>
}

// CHECK: rock.conv_elementwise_gemm
func.func @conv_gemm_with_bias_only(%arg0: tensor<131072xf32>, %arg1: tensor<36864xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<64xf32>) -> tensor<32768xf32> attributes {kernel, arch = ""} {
  %0 = tosa.const_shape  {values = dense<[1, 16, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xf32> into tensor<1x16x64xf32>
  %1 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1>} : (tensor<1x16x64xf32>) -> tensor<1x64x16xf32>
  %2 = tosa.const_shape  {values = dense<[64, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
  %3 = tosa.transpose %expanded_0 {perms = array<i32: 2, 1, 0>} : (tensor<64x1x1xf32>) -> tensor<1x1x64xf32>
  %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2048x64xf32>}> : () -> tensor<1x2048x64xf32>
  %5 = tosa.add %3, %4 : (tensor<1x1x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %6 = tosa.const_shape  {values = dense<[64, 3, 3, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [64, 3, 3, 64] : tensor<36864xf32> into tensor<64x3x3x64xf32>
  %7 = tosa.transpose %expanded_1 {perms = array<i32: 0, 3, 1, 2>} : (tensor<64x3x3x64xf32>) -> tensor<64x64x3x3xf32>
  %8 = tosa.const_shape  {values = dense<[2, 32, 32, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 32, 32, 64] : tensor<131072xf32> into tensor<2x32x32x64xf32>
  %9 = tosa.transpose %expanded_2 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %10 = tosa.transpose %9 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %11 = tosa.transpose %7 {perms = array<i32: 0, 2, 3, 1>} : (tensor<64x64x3x3xf32>) -> tensor<64x3x3x64xf32>
  %12 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %13 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %14 = tosa.conv2d %10, %11, %13, %12, %12 {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<2x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x64xf32>
  %15 = tosa.transpose %14 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %16 = tosa.transpose %15 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %17 = tosa.const_shape  {values = dense<[1, 2048, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed = tensor.collapse_shape %16 [[0, 1, 2], [3]] : tensor<2x32x32x64xf32> into tensor<2048x64xf32>
  %expanded_3 = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [1, 2048, 64] : tensor<2048x64xf32> into tensor<1x2048x64xf32>
  %18 = tosa.add %expanded_3, %5 : (tensor<1x2048x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %19 = tosa.matmul %18, %1, %12, %12 : (tensor<1x2048x64xf32>, tensor<1x64x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x16xf32>
  %20 = tosa.const_shape  {values = dense<32768> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_4 = tensor.collapse_shape %19 [[0, 1, 2]] : tensor<1x2048x16xf32> into tensor<32768xf32>
  return %collapsed_4 : tensor<32768xf32>
}

// CHECK: rock.conv_elementwise_gemm
func.func @conv_gemm_with_scale_and_bias(%arg0: tensor<131072xf32>, %arg1: tensor<36864xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<32768xf32> attributes {kernel, arch = ""} {
  %0 = tosa.const_shape  {values = dense<[1, 16, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xf32> into tensor<1x16x64xf32>
  %1 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1>} : (tensor<1x16x64xf32>) -> tensor<1x64x16xf32>
  %2 = tosa.const_shape  {values = dense<[64, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded_0 = tensor.expand_shape %arg4 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
  %3 = tosa.transpose %expanded_0 {perms = array<i32: 2, 1, 0>} : (tensor<64x1x1xf32>) -> tensor<1x1x64xf32>
  %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2048x64xf32>}> : () -> tensor<1x2048x64xf32>
  %5 = tosa.add %3, %4 : (tensor<1x1x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %expanded_1 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
  %6 = tosa.transpose %expanded_1 {perms = array<i32: 2, 1, 0>} : (tensor<64x1x1xf32>) -> tensor<1x1x64xf32>
  %7 = tosa.add %6, %4 : (tensor<1x1x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %8 = tosa.const_shape  {values = dense<[64, 3, 3, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [64, 3, 3, 64] : tensor<36864xf32> into tensor<64x3x3x64xf32>
  %9 = tosa.transpose %expanded_2 {perms = array<i32: 0, 3, 1, 2>} : (tensor<64x3x3x64xf32>) -> tensor<64x64x3x3xf32>
  %10 = tosa.const_shape  {values = dense<[2, 32, 32, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_3 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 32, 32, 64] : tensor<131072xf32> into tensor<2x32x32x64xf32>
  %11 = tosa.transpose %expanded_3 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %12 = tosa.transpose %11 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %13 = tosa.transpose %9 {perms = array<i32: 0, 2, 3, 1>} : (tensor<64x64x3x3xf32>) -> tensor<64x3x3x64xf32>
  %14 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %15 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %16 = tosa.conv2d %12, %13, %15, %14, %14 {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<2x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x64xf32>
  %17 = tosa.transpose %16 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %18 = tosa.transpose %17 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %19 = tosa.const_shape  {values = dense<[1, 2048, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed = tensor.collapse_shape %18 [[0, 1, 2], [3]] : tensor<2x32x32x64xf32> into tensor<2048x64xf32>
  %expanded_4 = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [1, 2048, 64] : tensor<2048x64xf32> into tensor<1x2048x64xf32>
  %20 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %21 = tosa.mul %expanded_4, %7, %20 : (tensor<1x2048x64xf32>, tensor<1x2048x64xf32>, tensor<1xi8>) -> tensor<1x2048x64xf32>
  %22 = tosa.add %21, %5 : (tensor<1x2048x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %23 = tosa.matmul %22, %1, %14, %14 : (tensor<1x2048x64xf32>, tensor<1x64x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x16xf32>
  %24 = tosa.const_shape  {values = dense<32768> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_5 = tensor.collapse_shape %23 [[0, 1, 2]] : tensor<1x2048x16xf32> into tensor<32768xf32>
  return %collapsed_5 : tensor<32768xf32>
}

// CHECK: rock.conv_elementwise_gemm
func.func @conv_gemm_with_scale_bias_exp(%arg0: tensor<131072xf32>, %arg1: tensor<36864xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<32768xf32> attributes {kernel, arch = ""} {
  %0 = tosa.const_shape  {values = dense<[1, 16, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xf32> into tensor<1x16x64xf32>
  %1 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1>} : (tensor<1x16x64xf32>) -> tensor<1x64x16xf32>
  %2 = tosa.const_shape  {values = dense<[64, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded_0 = tensor.expand_shape %arg4 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
  %3 = tosa.transpose %expanded_0 {perms = array<i32: 2, 1, 0>} : (tensor<64x1x1xf32>) -> tensor<1x1x64xf32>
  %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2048x64xf32>}> : () -> tensor<1x2048x64xf32>
  %5 = tosa.add %3, %4 : (tensor<1x1x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %expanded_1 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [64, 1, 1] : tensor<64xf32> into tensor<64x1x1xf32>
  %6 = tosa.transpose %expanded_1 {perms = array<i32: 2, 1, 0>} : (tensor<64x1x1xf32>) -> tensor<1x1x64xf32>
  %7 = tosa.add %6, %4 : (tensor<1x1x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %8 = tosa.const_shape  {values = dense<[64, 3, 3, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [64, 3, 3, 64] : tensor<36864xf32> into tensor<64x3x3x64xf32>
  %9 = tosa.transpose %expanded_2 {perms = array<i32: 0, 3, 1, 2>} : (tensor<64x3x3x64xf32>) -> tensor<64x64x3x3xf32>
  %10 = tosa.const_shape  {values = dense<[2, 32, 32, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_3 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 32, 32, 64] : tensor<131072xf32> into tensor<2x32x32x64xf32>
  %11 = tosa.transpose %expanded_3 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %12 = tosa.transpose %11 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %13 = tosa.transpose %9 {perms = array<i32: 0, 2, 3, 1>} : (tensor<64x64x3x3xf32>) -> tensor<64x3x3x64xf32>
  %14 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %15 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %16 = tosa.conv2d %12, %13, %15, %14, %14 {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<2x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x64xf32>
  %17 = tosa.transpose %16 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x32x32x64xf32>) -> tensor<2x64x32x32xf32>
  %18 = tosa.transpose %17 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x64x32x32xf32>) -> tensor<2x32x32x64xf32>
  %19 = tosa.const_shape  {values = dense<[1, 2048, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed = tensor.collapse_shape %18 [[0, 1, 2], [3]] : tensor<2x32x32x64xf32> into tensor<2048x64xf32>
  %expanded_4 = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [1, 2048, 64] : tensor<2048x64xf32> into tensor<1x2048x64xf32>
  %20 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %21 = tosa.mul %expanded_4, %7, %20 : (tensor<1x2048x64xf32>, tensor<1x2048x64xf32>, tensor<1xi8>) -> tensor<1x2048x64xf32>
  %22 = tosa.add %21, %5 : (tensor<1x2048x64xf32>, tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %23 = tosa.exp %22 : (tensor<1x2048x64xf32>) -> tensor<1x2048x64xf32>
  %24 = tosa.matmul %23, %1, %14, %14 : (tensor<1x2048x64xf32>, tensor<1x64x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x16xf32>
  %25 = tosa.const_shape  {values = dense<32768> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_5 = tensor.collapse_shape %24 [[0, 1, 2]] : tensor<1x2048x16xf32> into tensor<32768xf32>
  return %collapsed_5 : tensor<32768xf32>
}
