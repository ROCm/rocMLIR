// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @attention_gqa_8_32
// CHECK: rock.attention
// CHECK: numHeadsKV = 8 : i32, numHeadsQ = 32 : i32
func.func @attention_gqa_8_32(%arg0: tensor<6144xf16>, %arg1: tensor<65536xf16>, %arg2: tensor<65536xf16>, %arg3: tensor<1xi32>) -> tensor<4096xf16> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<[[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]]]> : tensor<1x1x1x64xi32>}> : () -> tensor<1x1x1x64xi32>
  %1 = tosa.const_shape  {values = dense<4096> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape  {values = dense<[32, 64, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape  {values = dense<[32, 1, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x32x1x64xf32>}> : () -> tensor<1x32x1x64xf32>
  %5 = "tosa.const"() <{values = dense<1> : tensor<1x32x1x64xi8>}> : () -> tensor<1x32x1x64xi8>
  %6 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
  %7 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x1x64xf16>}> : () -> tensor<1x32x1x64xf16>
  %8 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x1x64xf16>}> : () -> tensor<1x32x1x64xf16>
  %9 = tosa.const_shape  {values = dense<[1, 32, 1, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %10 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %11 = tosa.const_shape  {values = dense<[32, 128, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %12 = tosa.const_shape  {values = dense<[32, 1, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %13 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x8x4x128x64xf16>}> : () -> tensor<1x8x4x128x64xf16>
  %14 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x8x4x64x128xf16>}> : () -> tensor<1x8x4x64x128xf16>
  %15 = tosa.const_shape  {values = dense<[1, 8, 1, 64, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %16 = tosa.const_shape  {values = dense<[1, 32, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %17 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %18 = "tosa.const"() <{values = dense<1> : tensor<1x1x1x64xi32>}> : () -> tensor<1x1x1x64xi32>
  %19 = tosa.const_shape  {values = dense<[1, 48, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 48, 1, 128] : tensor<6144xf16> into tensor<1x48x1x128xf16>
  %extracted_slice = tensor.extract_slice %expanded[0, 0, 0, 0] [1, 32, 1, 128] [1, 1, 1, 1] : tensor<1x48x1x128xf16> to tensor<1x32x1x128xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [1, 8, 1, 64, 128] : tensor<65536xf16> into tensor<1x8x1x64x128xf16>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [1, 8, 1, 64, 128] : tensor<65536xf16> into tensor<1x8x1x64x128xf16>
  %20 = tosa.mul %expanded_1, %14, %17 : (tensor<1x8x1x64x128xf16>, tensor<1x8x4x64x128xf16>, tensor<1xi8>) -> tensor<1x8x4x64x128xf16>
  %21 = tosa.transpose %expanded_0 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<1x8x1x64x128xf16>) -> tensor<1x8x1x128x64xf16>
  %22 = tosa.mul %21, %13, %17 : (tensor<1x8x1x128x64xf16>, tensor<1x8x4x128x64xf16>, tensor<1xi8>) -> tensor<1x8x4x128x64xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x1x128xf16> into tensor<32x1x128xf16>
  %collapsed_2 = tensor.collapse_shape %22 [[0, 1, 2], [3], [4]] : tensor<1x8x4x128x64xf16> into tensor<32x128x64xf16>
  %23 = tosa.matmul %collapsed, %collapsed_2, %10, %10 {acc_type = f32} : (tensor<32x1x128xf16>, tensor<32x128x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x1x64xf16>
  %expanded_3 = tensor.expand_shape %23 [[0, 1], [2], [3]] output_shape [1, 32, 1, 64] : tensor<32x1x64xf16> into tensor<1x32x1x64xf16>
  %24 = tosa.mul %expanded_3, %7, %17 : (tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16>, tensor<1xi8>) -> tensor<1x32x1x64xf16>
  %expanded_4 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %25 = tosa.mul %expanded_4, %18, %17 : (tensor<1x1x1x1xi32>, tensor<1x1x1x64xi32>, tensor<1xi8>) -> tensor<1x1x1x64xi32>
  %26 = tosa.greater %0, %25 : (tensor<1x1x1x64xi32>, tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xi1>
  %27 = tosa.cast %26 : (tensor<1x1x1x64xi1>) -> tensor<1x1x1x64xi32>
  %28 = tosa.cast %27 : (tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xi8>
  %29 = tosa.mul %28, %5, %17 : (tensor<1x1x1x64xi8>, tensor<1x32x1x64xi8>, tensor<1xi8>) -> tensor<1x32x1x64xi8>
  %30 = tosa.cast %29 : (tensor<1x32x1x64xi8>) -> tensor<1x32x1x64xi1>
  %31 = tosa.select %30, %8, %24 : (tensor<1x32x1x64xi1>, tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16>) -> tensor<1x32x1x64xf16>
  %32 = tosa.cast %31 : (tensor<1x32x1x64xf16>) -> tensor<1x32x1x64xf32>
  %33 = tosa.reduce_max %32 {axis = 3 : i32} : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
  %34 = tosa.mul %33, %4, %17 : (tensor<1x32x1x1xf32>, tensor<1x32x1x64xf32>, tensor<1xi8>) -> tensor<1x32x1x64xf32>
  %35 = tosa.sub %32, %34 : (tensor<1x32x1x64xf32>, tensor<1x32x1x64xf32>) -> tensor<1x32x1x64xf32>
  %36 = tosa.exp %35 : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x64xf32>
  %37 = tosa.reduce_sum %36 {axis = 3 : i32} : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x1xf32>
  %38 = tosa.mul %37, %4, %17 : (tensor<1x32x1x1xf32>, tensor<1x32x1x64xf32>, tensor<1xi8>) -> tensor<1x32x1x64xf32>
  %39 = tosa.reciprocal %38 : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x64xf32>
  %40 = tosa.mul %36, %39, %17 : (tensor<1x32x1x64xf32>, tensor<1x32x1x64xf32>, tensor<1xi8>) -> tensor<1x32x1x64xf32>
  %41 = tosa.cast %40 : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x64xf16>
  %collapsed_5 = tensor.collapse_shape %41 [[0, 1], [2], [3]] : tensor<1x32x1x64xf16> into tensor<32x1x64xf16>
  %collapsed_6 = tensor.collapse_shape %20 [[0, 1, 2], [3], [4]] : tensor<1x8x4x64x128xf16> into tensor<32x64x128xf16>
  %42 = tosa.matmul %collapsed_5, %collapsed_6, %10, %10 {acc_type = f32} : (tensor<32x1x64xf16>, tensor<32x64x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x1x128xf16>
  %expanded_7 = tensor.expand_shape %42 [[0, 1], [2], [3]] output_shape [1, 32, 1, 128] : tensor<32x1x128xf16> into tensor<1x32x1x128xf16>
  %43 = tosa.transpose %expanded_7 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x32x1x128xf16>) -> tensor<1x1x32x128xf16>
  %collapsed_8 = tensor.collapse_shape %43 [[0, 1, 2, 3]] : tensor<1x1x32x128xf16> into tensor<4096xf16>
  return %collapsed_8 : tensor<4096xf16>
}


// CHECK-LABEL: func @attention_gqa_2_14
// CHECK: rock.attention
// CHECK: numHeadsKV = 2 : i32, numHeadsQ = 14 : i32
func.func @attention_gqa_2_14(%arg0: tensor<2xi32> {mhal.read_access}, %arg1: tensor<9216xf16> {mhal.read_access}, %arg2: tensor<2048xf16> {mhal.read_access}, %arg3: tensor<2048xf16> {mhal.read_access}) -> (tensor<7168xf16> {mhal.write_access}) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<[[[[0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]]]> : tensor<1x1x4x8xi8>}> : () -> tensor<1x1x4x8xi8>
  %1 = "tosa.const"() <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  %2 = tosa.const_shape  {values = dense<7168> : tensor<1xindex>} : () -> !tosa.shape<1>
  %3 = tosa.const_shape  {values = dense<[28, 8, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<[28, 4, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %5 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x14x4x8xf32>}> : () -> tensor<2x14x4x8xf32>
  %6 = "tosa.const"() <{values = dense<0xFC00> : tensor<2x14x4x8xf16>}> : () -> tensor<2x14x4x8xf16>
  %7 = tosa.const_shape  {values = dense<[2, 14, 4, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %8 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %9 = tosa.const_shape  {values = dense<[28, 64, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %10 = tosa.const_shape  {values = dense<[28, 4, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %11 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x64x8xf16>}> : () -> tensor<2x2x7x64x8xf16>
  %12 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x8x64xf16>}> : () -> tensor<2x2x7x8x64xf16>
  %13 = tosa.const_shape  {values = dense<[2, 2, 1, 8, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %14 = tosa.const_shape  {values = dense<[2, 14, 4, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %15 = tosa.const_shape  {values = dense<[2, 1, 1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %16 = "tosa.const"() <{values = dense<1> : tensor<2x14x4x8xi8>}> : () -> tensor<2x14x4x8xi8>
  %17 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<2x14x4x8xf16>}> : () -> tensor<2x14x4x8xf16>
  %18 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %19 = "tosa.const"() <{values = dense<1> : tensor<2x8xi32>}> : () -> tensor<2x8xi32>
  %20 = tosa.const_shape  {values = dense<[2, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %21 = tosa.const_shape  {values = dense<[2, 4, 18, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 4, 18, 64] : tensor<9216xf16> into tensor<2x4x18x64xf16>
  %22 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x4x18x64xf16>) -> tensor<2x18x4x64xf16>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
  %23 = tosa.mul %1, %19, %18 : (tensor<1x8xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
  %24 = tosa.mul %0, %16, %18 : (tensor<1x1x4x8xi8>, tensor<2x14x4x8xi8>, tensor<1xi8>) -> tensor<2x14x4x8xi8>
  %25 = tosa.mul %expanded_0, %19, %18 : (tensor<2x1xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
  %26 = tosa.greater %23, %25 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
  %27 = tosa.cast %26 : (tensor<2x8xi1>) -> tensor<2x8xi32>
  %28 = tosa.cast %27 : (tensor<2x8xi32>) -> tensor<2x8xi8>
  %expanded_1 = tensor.expand_shape %28 [[0, 1, 2], [3]] output_shape [2, 1, 1, 8] : tensor<2x8xi8> into tensor<2x1x1x8xi8>
  %29 = tosa.mul %expanded_1, %16, %18 : (tensor<2x1x1x8xi8>, tensor<2x14x4x8xi8>, tensor<1xi8>) -> tensor<2x14x4x8xi8>
  %extracted_slice = tensor.extract_slice %22[0, 0, 0, 0] [2, 14, 4, 64] [1, 1, 1, 1] : tensor<2x18x4x64xf16> to tensor<2x14x4x64xf16>
  %expanded_2 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
  %30 = tosa.mul %expanded_2, %12, %18 : (tensor<2x2x1x8x64xf16>, tensor<2x2x7x8x64xf16>, tensor<1xi8>) -> tensor<2x2x7x8x64xf16>
  %expanded_3 = tensor.expand_shape %arg3 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
  %31 = tosa.transpose %expanded_3 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<2x2x1x8x64xf16>) -> tensor<2x2x1x64x8xf16>
  %32 = tosa.mul %31, %11, %18 : (tensor<2x2x1x64x8xf16>, tensor<2x2x7x64x8xf16>, tensor<1xi8>) -> tensor<2x2x7x64x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<2x14x4x64xf16> into tensor<28x4x64xf16>
  %collapsed_4 = tensor.collapse_shape %32 [[0, 1, 2], [3], [4]] : tensor<2x2x7x64x8xf16> into tensor<28x64x8xf16>
  %33 = tosa.matmul %collapsed, %collapsed_4, %8, %8 {acc_type = f32} : (tensor<28x4x64xf16>, tensor<28x64x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x4x8xf16>
  %expanded_5 = tensor.expand_shape %33 [[0, 1], [2], [3]] output_shape [2, 14, 4, 8] : tensor<28x4x8xf16> into tensor<2x14x4x8xf16>
  %34 = tosa.mul %expanded_5, %17, %18 : (tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>, tensor<1xi8>) -> tensor<2x14x4x8xf16>
  %35 = tosa.cast %24 : (tensor<2x14x4x8xi8>) -> tensor<2x14x4x8xi1>
  %36 = tosa.select %35, %6, %34 : (tensor<2x14x4x8xi1>, tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf16>
  %37 = tosa.cast %29 : (tensor<2x14x4x8xi8>) -> tensor<2x14x4x8xi1>
  %38 = tosa.select %37, %6, %36 : (tensor<2x14x4x8xi1>, tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf16>
  %39 = tosa.cast %38 : (tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf32>
  %40 = tosa.reduce_max %39 {axis = 3 : i32} : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x1xf32>
  %41 = tosa.mul %40, %5, %18 : (tensor<2x14x4x1xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
  %42 = tosa.sub %39, %41 : (tensor<2x14x4x8xf32>, tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
  %43 = tosa.exp %42 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
  %44 = tosa.reduce_sum %43 {axis = 3 : i32} : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x1xf32>
  %45 = tosa.mul %44, %5, %18 : (tensor<2x14x4x1xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
  %46 = tosa.reciprocal %45 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
  %47 = tosa.mul %43, %46, %18 : (tensor<2x14x4x8xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
  %48 = tosa.cast %47 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf16>
  %collapsed_6 = tensor.collapse_shape %48 [[0, 1], [2], [3]] : tensor<2x14x4x8xf16> into tensor<28x4x8xf16>
  %collapsed_7 = tensor.collapse_shape %30 [[0, 1, 2], [3], [4]] : tensor<2x2x7x8x64xf16> into tensor<28x8x64xf16>
  %49 = tosa.matmul %collapsed_6, %collapsed_7, %8, %8 {acc_type = f32} : (tensor<28x4x8xf16>, tensor<28x8x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x4x64xf16>
  %expanded_8 = tensor.expand_shape %49 [[0, 1], [2], [3]] output_shape [2, 14, 4, 64] : tensor<28x4x64xf16> into tensor<2x14x4x64xf16>
  %50 = tosa.transpose %expanded_8 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x14x4x64xf16>) -> tensor<2x4x14x64xf16>
  %collapsed_9 = tensor.collapse_shape %50 [[0, 1, 2, 3]] : tensor<2x4x14x64xf16> into tensor<7168xf16>
  return %collapsed_9 : tensor<7168xf16>
}

// CHECK-LABEL: func @attention_gqa_2_14_2
// CHECK: rock.attention
// CHECK: numHeadsKV = 2 : i32, numHeadsQ = 14 : i32
func.func @attention_gqa_2_14_2(%arg0: tensor<9216xf16> {mhal.read_access}, %arg1: tensor<2048xf16> {mhal.read_access}, %arg2: tensor<2048xf16> {mhal.read_access}, %arg3: tensor<2xi32> {mhal.read_access}) -> (tensor<7168xf16> {mhal.write_access}) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<[[[[0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]]]> : tensor<1x1x4x8xi8>}> : () -> tensor<1x1x4x8xi8>
  %1 = "tosa.const"() <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  %2 = tosa.const_shape  {values = dense<7168> : tensor<1xindex>} : () -> !tosa.shape<1>
  %3 = tosa.const_shape  {values = dense<[28, 8, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<[28, 4, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %5 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x14x4x8xf32>}> : () -> tensor<2x14x4x8xf32>
  %6 = tosa.const_shape  {values = dense<[2, 1, 1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %7 = "tosa.const"() <{values = dense<1> : tensor<2x14x4x8xi8>}> : () -> tensor<2x14x4x8xi8>
  %8 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<2x14x4x8xf16>}> : () -> tensor<2x14x4x8xf16>
  %9 = "tosa.const"() <{values = dense<0xFC00> : tensor<2x14x4x8xf16>}> : () -> tensor<2x14x4x8xf16>
  %10 = "tosa.const"() <{values = dense<1> : tensor<2x8xi32>}> : () -> tensor<2x8xi32>
  %11 = tosa.const_shape  {values = dense<[2, 14, 4, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %12 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %13 = tosa.const_shape  {values = dense<[28, 64, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %14 = tosa.const_shape  {values = dense<[28, 4, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %15 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x64x8xf16>}> : () -> tensor<2x2x7x64x8xf16>
  %16 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %17 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x8x64xf16>}> : () -> tensor<2x2x7x8x64xf16>
  %18 = tosa.const_shape  {values = dense<[2, 2, 1, 8, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %19 = tosa.const_shape  {values = dense<[2, 14, 4, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %20 = tosa.const_shape  {values = dense<[2, 4, 18, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %21 = tosa.const_shape  {values = dense<[2, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %expanded = tensor.expand_shape %arg3 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 4, 18, 64] : tensor<9216xf16> into tensor<2x4x18x64xf16>
  %22 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x4x18x64xf16>) -> tensor<2x18x4x64xf16>
  %extracted_slice = tensor.extract_slice %22[0, 0, 0, 0] [2, 14, 4, 64] [1, 1, 1, 1] : tensor<2x18x4x64xf16> to tensor<2x14x4x64xf16>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
  %23 = tosa.mul %expanded_1, %17, %16 : (tensor<2x2x1x8x64xf16>, tensor<2x2x7x8x64xf16>, tensor<1xi8>) -> tensor<2x2x7x8x64xf16>
  %expanded_2 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
  %24 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<2x2x1x8x64xf16>) -> tensor<2x2x1x64x8xf16>
  %25 = tosa.mul %24, %15, %16 : (tensor<2x2x1x64x8xf16>, tensor<2x2x7x64x8xf16>, tensor<1xi8>) -> tensor<2x2x7x64x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<2x14x4x64xf16> into tensor<28x4x64xf16>
  %collapsed_3 = tensor.collapse_shape %25 [[0, 1, 2], [3], [4]] : tensor<2x2x7x64x8xf16> into tensor<28x64x8xf16>
  %26 = tosa.matmul %collapsed, %collapsed_3, %12, %12 {acc_type = f32} : (tensor<28x4x64xf16>, tensor<28x64x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x4x8xf16>
  %expanded_4 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [2, 14, 4, 8] : tensor<28x4x8xf16> into tensor<2x14x4x8xf16>
  %27 = tosa.mul %1, %10, %16 : (tensor<1x8xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
  %28 = tosa.mul %expanded_4, %8, %16 : (tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>, tensor<1xi8>) -> tensor<2x14x4x8xf16>
  %29 = tosa.mul %0, %7, %16 : (tensor<1x1x4x8xi8>, tensor<2x14x4x8xi8>, tensor<1xi8>) -> tensor<2x14x4x8xi8>
  %30 = tosa.cast %29 : (tensor<2x14x4x8xi8>) -> tensor<2x14x4x8xi1>
  %31 = tosa.select %30, %9, %28 : (tensor<2x14x4x8xi1>, tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf16>
  %32 = tosa.mul %expanded, %10, %16 : (tensor<2x1xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
  %33 = tosa.greater %27, %32 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
  %34 = tosa.cast %33 : (tensor<2x8xi1>) -> tensor<2x8xi32>
  %35 = tosa.cast %34 : (tensor<2x8xi32>) -> tensor<2x8xi8>
  %expanded_5 = tensor.expand_shape %35 [[0, 1, 2], [3]] output_shape [2, 1, 1, 8] : tensor<2x8xi8> into tensor<2x1x1x8xi8>
  %36 = tosa.mul %expanded_5, %7, %16 : (tensor<2x1x1x8xi8>, tensor<2x14x4x8xi8>, tensor<1xi8>) -> tensor<2x14x4x8xi8>
  %37 = tosa.cast %36 : (tensor<2x14x4x8xi8>) -> tensor<2x14x4x8xi1>
  %38 = tosa.select %37, %9, %31 : (tensor<2x14x4x8xi1>, tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf16>
  %39 = tosa.cast %38 : (tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf32>
  %40 = tosa.reduce_max %39 {axis = 3 : i32} : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x1xf32>
  %41 = tosa.mul %40, %5, %16 : (tensor<2x14x4x1xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
  %42 = tosa.sub %39, %41 : (tensor<2x14x4x8xf32>, tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
  %43 = tosa.exp %42 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
  %44 = tosa.reduce_sum %43 {axis = 3 : i32} : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x1xf32>
  %45 = tosa.mul %44, %5, %16 : (tensor<2x14x4x1xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
  %46 = tosa.reciprocal %45 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
  %47 = tosa.mul %43, %46, %16 : (tensor<2x14x4x8xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
  %48 = tosa.cast %47 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf16>
  %collapsed_6 = tensor.collapse_shape %48 [[0, 1], [2], [3]] : tensor<2x14x4x8xf16> into tensor<28x4x8xf16>
  %collapsed_7 = tensor.collapse_shape %23 [[0, 1, 2], [3], [4]] : tensor<2x2x7x8x64xf16> into tensor<28x8x64xf16>
  %49 = tosa.matmul %collapsed_6, %collapsed_7, %12, %12 {acc_type = f32} : (tensor<28x4x8xf16>, tensor<28x8x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x4x64xf16>
  %expanded_8 = tensor.expand_shape %49 [[0, 1], [2], [3]] output_shape [2, 14, 4, 64] : tensor<28x4x64xf16> into tensor<2x14x4x64xf16>
  %50 = tosa.transpose %expanded_8 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x14x4x64xf16>) -> tensor<2x4x14x64xf16>
  %collapsed_9 = tensor.collapse_shape %50 [[0, 1, 2, 3]] : tensor<2x4x14x64xf16> into tensor<7168xf16>
  return %collapsed_9 : tensor<7168xf16>
}

// CHECK-LABEL: func @attention_gqa_2_14_3
// CHECK: rock.attention
// CHECK: numHeadsKV = 2 : i32, numHeadsQ = 14 : i32
func.func @attention_gqa_2_14_3(%arg0: tensor<2304xf16> {mhal.read_access}, %arg1: tensor<2048xf16> {mhal.read_access}, %arg2: tensor<2048xf16> {mhal.read_access}, %arg3: tensor<2xi32> {mhal.read_access}) -> (tensor<1792xf16> {mhal.write_access}) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  %1 = tosa.const_shape  {values = dense<1792> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape  {values = dense<[28, 8, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape  {values = dense<[28, 1, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x14x1x8xf32>}> : () -> tensor<2x14x1x8xf32>
  %5 = "tosa.const"() <{values = dense<1> : tensor<2x14x1x8xi8>}> : () -> tensor<2x14x1x8xi8>
  %6 = tosa.const_shape  {values = dense<[2, 1, 1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %7 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<2x14x1x8xf16>}> : () -> tensor<2x14x1x8xf16>
  %8 = "tosa.const"() <{values = dense<0xFC00> : tensor<2x14x1x8xf16>}> : () -> tensor<2x14x1x8xf16>
  %9 = "tosa.const"() <{values = dense<1> : tensor<2x8xi32>}> : () -> tensor<2x8xi32>
  %10 = tosa.const_shape  {values = dense<[2, 14, 1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %11 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %12 = tosa.const_shape  {values = dense<[28, 64, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %13 = tosa.const_shape  {values = dense<[28, 1, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %14 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x64x8xf16>}> : () -> tensor<2x2x7x64x8xf16>
  %15 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %16 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x8x64xf16>}> : () -> tensor<2x2x7x8x64xf16>
  %17 = tosa.const_shape  {values = dense<[2, 2, 1, 8, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %18 = tosa.const_shape  {values = dense<[2, 14, 1, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %19 = tosa.const_shape  {values = dense<[2, 18, 1, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %20 = tosa.const_shape  {values = dense<[2, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %expanded = tensor.expand_shape %arg3 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 18, 1, 64] : tensor<2304xf16> into tensor<2x18x1x64xf16>
  %extracted_slice = tensor.extract_slice %expanded_0[0, 0, 0, 0] [2, 14, 1, 64] [1, 1, 1, 1] : tensor<2x18x1x64xf16> to tensor<2x14x1x64xf16>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
  %21 = tosa.mul %expanded_1, %16, %15 : (tensor<2x2x1x8x64xf16>, tensor<2x2x7x8x64xf16>, tensor<1xi8>) -> tensor<2x2x7x8x64xf16>
  %expanded_2 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
  %22 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<2x2x1x8x64xf16>) -> tensor<2x2x1x64x8xf16>
  %23 = tosa.mul %22, %14, %15 : (tensor<2x2x1x64x8xf16>, tensor<2x2x7x64x8xf16>, tensor<1xi8>) -> tensor<2x2x7x64x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<2x14x1x64xf16> into tensor<28x1x64xf16>
  %collapsed_3 = tensor.collapse_shape %23 [[0, 1, 2], [3], [4]] : tensor<2x2x7x64x8xf16> into tensor<28x64x8xf16>
  %24 = tosa.matmul %collapsed, %collapsed_3, %11, %11 {acc_type = f32} : (tensor<28x1x64xf16>, tensor<28x64x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x1x8xf16>
  %expanded_4 = tensor.expand_shape %24 [[0, 1], [2], [3]] output_shape [2, 14, 1, 8] : tensor<28x1x8xf16> into tensor<2x14x1x8xf16>
  %25 = tosa.mul %0, %9, %15 : (tensor<1x8xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
  %26 = tosa.mul %expanded_4, %7, %15 : (tensor<2x14x1x8xf16>, tensor<2x14x1x8xf16>, tensor<1xi8>) -> tensor<2x14x1x8xf16>
  %27 = tosa.mul %expanded, %9, %15 : (tensor<2x1xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
  %28 = tosa.greater %25, %27 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
  %29 = tosa.cast %28 : (tensor<2x8xi1>) -> tensor<2x8xi32>
  %30 = tosa.cast %29 : (tensor<2x8xi32>) -> tensor<2x8xi8>
  %expanded_5 = tensor.expand_shape %30 [[0, 1, 2], [3]] output_shape [2, 1, 1, 8] : tensor<2x8xi8> into tensor<2x1x1x8xi8>
  %31 = tosa.mul %expanded_5, %5, %15 : (tensor<2x1x1x8xi8>, tensor<2x14x1x8xi8>, tensor<1xi8>) -> tensor<2x14x1x8xi8>
  %32 = tosa.cast %31 : (tensor<2x14x1x8xi8>) -> tensor<2x14x1x8xi1>
  %33 = tosa.select %32, %8, %26 : (tensor<2x14x1x8xi1>, tensor<2x14x1x8xf16>, tensor<2x14x1x8xf16>) -> tensor<2x14x1x8xf16>
  %34 = tosa.cast %33 : (tensor<2x14x1x8xf16>) -> tensor<2x14x1x8xf32>
  %35 = tosa.reduce_max %34 {axis = 3 : i32} : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x1xf32>
  %36 = tosa.mul %35, %4, %15 : (tensor<2x14x1x1xf32>, tensor<2x14x1x8xf32>, tensor<1xi8>) -> tensor<2x14x1x8xf32>
  %37 = tosa.sub %34, %36 : (tensor<2x14x1x8xf32>, tensor<2x14x1x8xf32>) -> tensor<2x14x1x8xf32>
  %38 = tosa.exp %37 : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x8xf32>
  %39 = tosa.reduce_sum %38 {axis = 3 : i32} : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x1xf32>
  %40 = tosa.mul %39, %4, %15 : (tensor<2x14x1x1xf32>, tensor<2x14x1x8xf32>, tensor<1xi8>) -> tensor<2x14x1x8xf32>
  %41 = tosa.reciprocal %40 : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x8xf32>
  %42 = tosa.mul %38, %41, %15 : (tensor<2x14x1x8xf32>, tensor<2x14x1x8xf32>, tensor<1xi8>) -> tensor<2x14x1x8xf32>
  %43 = tosa.cast %42 : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x8xf16>
  %collapsed_6 = tensor.collapse_shape %43 [[0, 1], [2], [3]] : tensor<2x14x1x8xf16> into tensor<28x1x8xf16>
  %collapsed_7 = tensor.collapse_shape %21 [[0, 1, 2], [3], [4]] : tensor<2x2x7x8x64xf16> into tensor<28x8x64xf16>
  %44 = tosa.matmul %collapsed_6, %collapsed_7, %11, %11 {acc_type = f32} : (tensor<28x1x8xf16>, tensor<28x8x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x1x64xf16>
  %expanded_8 = tensor.expand_shape %44 [[0, 1], [2], [3]] output_shape [2, 14, 1, 64] : tensor<28x1x64xf16> into tensor<2x14x1x64xf16>
  %45 = tosa.transpose %expanded_8 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x14x1x64xf16>) -> tensor<2x1x14x64xf16>
  %collapsed_9 = tensor.collapse_shape %45 [[0, 1, 2, 3]] : tensor<2x1x14x64xf16> into tensor<1792xf16>
  return %collapsed_9 : tensor<1792xf16>
}

// CHECK-LABEL: func @attention_gqa_2_14_4
// CHECK: rock.attention
// CHECK: numHeadsKV = 2 : i32, numHeadsQ = 14 : i32
func.func @attention_gqa_2_14_4(%arg0: tensor<2xi32> {mhal.read_access}, %arg1: tensor<2304xf16> {mhal.read_access}, %arg2: tensor<2048xf16> {mhal.read_access}, %arg3: tensor<2048xf16> {mhal.read_access}) -> (tensor<1792xf16> {mhal.write_access}) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  %1 = tosa.const_shape  {values = dense<1792> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape  {values = dense<[28, 8, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape  {values = dense<[28, 1, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x14x1x8xf32>}> : () -> tensor<2x14x1x8xf32>
  %5 = "tosa.const"() <{values = dense<0xFC00> : tensor<2x14x1x8xf16>}> : () -> tensor<2x14x1x8xf16>
  %6 = tosa.const_shape  {values = dense<[2, 14, 1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %8 = tosa.const_shape  {values = dense<[28, 64, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %9 = tosa.const_shape  {values = dense<[28, 1, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %10 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x64x8xf16>}> : () -> tensor<2x2x7x64x8xf16>
  %11 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x8x64xf16>}> : () -> tensor<2x2x7x8x64xf16>
  %12 = tosa.const_shape  {values = dense<[2, 2, 1, 8, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %13 = tosa.const_shape  {values = dense<[2, 14, 1, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %14 = "tosa.const"() <{values = dense<1> : tensor<2x14x1x8xi8>}> : () -> tensor<2x14x1x8xi8>
  %15 = tosa.const_shape  {values = dense<[2, 1, 1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %16 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<2x14x1x8xf16>}> : () -> tensor<2x14x1x8xf16>
  %17 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %18 = "tosa.const"() <{values = dense<1> : tensor<2x8xi32>}> : () -> tensor<2x8xi32>
  %19 = tosa.const_shape  {values = dense<[2, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %20 = tosa.const_shape  {values = dense<[2, 18, 1, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 18, 1, 64] : tensor<2304xf16> into tensor<2x18x1x64xf16>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
  %21 = tosa.mul %0, %18, %17 : (tensor<1x8xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
  %22 = tosa.mul %expanded_0, %18, %17 : (tensor<2x1xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
  %23 = tosa.greater %21, %22 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
  %24 = tosa.cast %23 : (tensor<2x8xi1>) -> tensor<2x8xi32>
  %25 = tosa.cast %24 : (tensor<2x8xi32>) -> tensor<2x8xi8>
  %expanded_1 = tensor.expand_shape %25 [[0, 1, 2], [3]] output_shape [2, 1, 1, 8] : tensor<2x8xi8> into tensor<2x1x1x8xi8>
  %26 = tosa.mul %expanded_1, %14, %17 : (tensor<2x1x1x8xi8>, tensor<2x14x1x8xi8>, tensor<1xi8>) -> tensor<2x14x1x8xi8>
  %extracted_slice = tensor.extract_slice %expanded[0, 0, 0, 0] [2, 14, 1, 64] [1, 1, 1, 1] : tensor<2x18x1x64xf16> to tensor<2x14x1x64xf16>
  %expanded_2 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
  %27 = tosa.mul %expanded_2, %11, %17 : (tensor<2x2x1x8x64xf16>, tensor<2x2x7x8x64xf16>, tensor<1xi8>) -> tensor<2x2x7x8x64xf16>
  %expanded_3 = tensor.expand_shape %arg3 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
  %28 = tosa.transpose %expanded_3 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<2x2x1x8x64xf16>) -> tensor<2x2x1x64x8xf16>
  %29 = tosa.mul %28, %10, %17 : (tensor<2x2x1x64x8xf16>, tensor<2x2x7x64x8xf16>, tensor<1xi8>) -> tensor<2x2x7x64x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<2x14x1x64xf16> into tensor<28x1x64xf16>
  %collapsed_4 = tensor.collapse_shape %29 [[0, 1, 2], [3], [4]] : tensor<2x2x7x64x8xf16> into tensor<28x64x8xf16>
  %30 = tosa.matmul %collapsed, %collapsed_4, %7, %7 {acc_type = f32} : (tensor<28x1x64xf16>, tensor<28x64x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x1x8xf16>
  %expanded_5 = tensor.expand_shape %30 [[0, 1], [2], [3]] output_shape [2, 14, 1, 8] : tensor<28x1x8xf16> into tensor<2x14x1x8xf16>
  %31 = tosa.mul %expanded_5, %16, %17 : (tensor<2x14x1x8xf16>, tensor<2x14x1x8xf16>, tensor<1xi8>) -> tensor<2x14x1x8xf16>
  %32 = tosa.cast %26 : (tensor<2x14x1x8xi8>) -> tensor<2x14x1x8xi1>
  %33 = tosa.select %32, %5, %31 : (tensor<2x14x1x8xi1>, tensor<2x14x1x8xf16>, tensor<2x14x1x8xf16>) -> tensor<2x14x1x8xf16>
  %34 = tosa.cast %33 : (tensor<2x14x1x8xf16>) -> tensor<2x14x1x8xf32>
  %35 = tosa.reduce_max %34 {axis = 3 : i32} : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x1xf32>
  %36 = tosa.mul %35, %4, %17 : (tensor<2x14x1x1xf32>, tensor<2x14x1x8xf32>, tensor<1xi8>) -> tensor<2x14x1x8xf32>
  %37 = tosa.sub %34, %36 : (tensor<2x14x1x8xf32>, tensor<2x14x1x8xf32>) -> tensor<2x14x1x8xf32>
  %38 = tosa.exp %37 : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x8xf32>
  %39 = tosa.reduce_sum %38 {axis = 3 : i32} : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x1xf32>
  %40 = tosa.mul %39, %4, %17 : (tensor<2x14x1x1xf32>, tensor<2x14x1x8xf32>, tensor<1xi8>) -> tensor<2x14x1x8xf32>
  %41 = tosa.reciprocal %40 : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x8xf32>
  %42 = tosa.mul %38, %41, %17 : (tensor<2x14x1x8xf32>, tensor<2x14x1x8xf32>, tensor<1xi8>) -> tensor<2x14x1x8xf32>
  %43 = tosa.cast %42 : (tensor<2x14x1x8xf32>) -> tensor<2x14x1x8xf16>
  %collapsed_6 = tensor.collapse_shape %43 [[0, 1], [2], [3]] : tensor<2x14x1x8xf16> into tensor<28x1x8xf16>
  %collapsed_7 = tensor.collapse_shape %27 [[0, 1, 2], [3], [4]] : tensor<2x2x7x8x64xf16> into tensor<28x8x64xf16>
  %44 = tosa.matmul %collapsed_6, %collapsed_7, %7, %7 {acc_type = f32} : (tensor<28x1x8xf16>, tensor<28x8x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x1x64xf16>
  %expanded_8 = tensor.expand_shape %44 [[0, 1], [2], [3]] output_shape [2, 14, 1, 64] : tensor<28x1x64xf16> into tensor<2x14x1x64xf16>
  %45 = tosa.transpose %expanded_8 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x14x1x64xf16>) -> tensor<2x1x14x64xf16>
  %collapsed_9 = tensor.collapse_shape %45 [[0, 1, 2, 3]] : tensor<2x1x14x64xf16> into tensor<1792xf16>
  return %collapsed_9 : tensor<1792xf16>
}
