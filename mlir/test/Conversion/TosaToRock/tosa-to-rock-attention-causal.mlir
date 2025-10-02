// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @mlir_nokvcache_causal_attention
// CHECK: rock.attention
// CHECK-NOT: currentSeqLen = 
// CHECK: causal
func.func @mlir_nokvcache_causal_attention(%arg0: tensor<24576xf16>, %arg1: tensor<262144xf16>, %arg2: tensor<262144xf16>) -> tensor<8192xf16> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 32, 64, 128] : tensor<262144xf16> into tensor<1x32x64x128xf16>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 32, 64, 128] : tensor<262144xf16> into tensor<1x32x64x128xf16>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 2, 96, 128] : tensor<24576xf16> into tensor<1x2x96x128xf16>
  %5 = tosa.transpose %expanded_2 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x96x128xf16>) -> tensor<1x96x2x128xf16>
  %6 = "tosa.const"() <{values = dense<[1, 2]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %7 = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi32>}> : () -> tensor<64xi32>
  %extracted_slice = tensor.extract_slice %5[0, 0, 0, 0] [1, 32, 2, 128] [1, 1, 1, 1] : tensor<1x96x2x128xf16> to tensor<1x32x2x128xf16>
  %8 = "tosa.const"() <{values = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %9 = tosa.transpose %expanded_1 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x32x64x128xf16>) -> tensor<1x32x128x64xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x2x128xf16> into tensor<32x2x128xf16>
  %collapsed_3 = tensor.collapse_shape %9 [[0, 1], [2], [3]] : tensor<1x32x128x64xf16> into tensor<32x128x64xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %10 = tosa.matmul %collapsed, %collapsed_3, %a_zp, %b_zp : (tensor<32x2x128xf16>, tensor<32x128x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x2x64xf16>
  %expanded_4 = tensor.expand_shape %10 [[0, 1], [2], [3]] output_shape [1, 32, 2, 64] : tensor<32x2x64xf16> into tensor<1x32x2x64xf16>
  %cst = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]]]> : tensor<1x1x1x64xi32>
  %11 = "tosa.const"() <{values = dense<1> : tensor<1x32x2x64xi32>}> : () -> tensor<1x32x2x64xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %12 = tosa.mul %cst, %11, %shift: (tensor<1x1x1x64xi32>, tensor<1x32x2x64xi32>, tensor<1xi8>) -> tensor<1x32x2x64xi32>
  %13 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %14 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %fused = tosa.mul %expanded_4, %14, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %cst_6 = arith.constant dense<[[[[0], [1]]]]> : tensor<1x1x2x1xi32>
  %15 = tosa.mul %cst_6, %11, %shift : (tensor<1x1x2x1xi32>, tensor<1x32x2x64xi32>, tensor<1xi8>) -> tensor<1x32x2x64xi32>
  %16 = tosa.greater %12, %15 : (tensor<1x32x2x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi1>
  %17 = tosa.cast %16 : (tensor<1x32x2x64xi1>) -> tensor<1x32x2x64xi32>
  %18 = tosa.cast %17 : (tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi8>
  %19 = tosa.cast %18 : (tensor<1x32x2x64xi8>) -> tensor<1x32x2x64xi1>
  %20 = tosa.select %19, %13, %fused : (tensor<1x32x2x64xi1>, tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %28 = tosa.reduce_max %20 {axis = 3 : i32} : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x1xf16>
  %29 = tosa.sub %20, %28 : (tensor<1x32x2x64xf16>, tensor<1x32x2x1xf16>) -> tensor<1x32x2x64xf16>
  %30 = tosa.exp %29 : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %31 = tosa.reduce_sum %30 {axis = 3 : i32} : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x1xf16>
  %32 = tosa.reciprocal %31 : (tensor<1x32x2x1xf16>) -> tensor<1x32x2x1xf16>
  %33 = tosa.mul %30, %32, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x1xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %collapsed_8 = tensor.collapse_shape %33 [[0, 1], [2], [3]] : tensor<1x32x2x64xf16> into tensor<32x2x64xf16>
  %expanded_9 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [32, 64, 128] : tensor<262144xf16> into tensor<32x64x128xf16>
  %34 = tosa.matmul %collapsed_8, %expanded_9, %a_zp, %b_zp : (tensor<32x2x64xf16>, tensor<32x64x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x2x128xf16>
  %expanded_10 = tensor.expand_shape %34 [[0, 1], [2], [3]] output_shape [1, 32, 2, 128] : tensor<32x2x128xf16> into tensor<1x32x2x128xf16>
  %35 = tosa.transpose %expanded_10 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x32x2x128xf16>) -> tensor<1x2x32x128xf16>
  %collapsed_11 = tensor.collapse_shape %35 [[0], [1], [2, 3]] : tensor<1x2x32x128xf16> into tensor<1x2x4096xf16>
  %collapsed_12 = tensor.collapse_shape %35 [[0, 1, 2, 3]] : tensor<1x2x32x128xf16> into tensor<8192xf16>
  return %collapsed_12 : tensor<8192xf16>
}

// CHECK-LABEL: func @mlir_causal_attention
// CHECK: rock.attention
// CHECK: currentSeqLen = (%{{.*}} : tensor<32xi32>)
// CHECK: causal
func.func @mlir_causal_attention(%arg0: tensor<24576xf16>, %arg1: tensor<262144xf16>, %arg2: tensor<262144xf16>, %arg3: tensor<1xi32>) -> tensor<8192xf16> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 32, 64, 128] : tensor<262144xf16> into tensor<1x32x64x128xf16>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
  %1 = tosa.transpose %expanded_0 {perms = array<i32: 1, 0>} : (tensor<1x1xi32>) -> tensor<1x1xi32>
  %2 = "tosa.const"() <{values = dense<1> : tensor<1x32xi32>}> : () -> tensor<1x32xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.mul %1, %2, %shift : (tensor<1x1xi32>, tensor<1x32xi32>, tensor<1xi8>) -> tensor<1x32xi32>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 32, 64, 128] : tensor<262144xf16> into tensor<1x32x64x128xf16>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 2, 96, 128] : tensor<24576xf16> into tensor<1x2x96x128xf16>
  %5 = tosa.transpose %expanded_2 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x96x128xf16>) -> tensor<1x96x2x128xf16>
  %6 = "tosa.const"() <{values = dense<[1, 2]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %7 = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi32>}> : () -> tensor<64xi32>
  %extracted_slice = tensor.extract_slice %5[0, 0, 0, 0] [1, 32, 2, 128] [1, 1, 1, 1] : tensor<1x96x2x128xf16> to tensor<1x32x2x128xf16>
  %8 = "tosa.const"() <{values = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %9 = tosa.transpose %expanded_1 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x32x64x128xf16>) -> tensor<1x32x128x64xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x2x128xf16> into tensor<32x2x128xf16>
  %collapsed_3 = tensor.collapse_shape %9 [[0, 1], [2], [3]] : tensor<1x32x128x64xf16> into tensor<32x128x64xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %10 = tosa.matmul %collapsed, %collapsed_3, %a_zp, %b_zp : (tensor<32x2x128xf16>, tensor<32x128x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x2x64xf16>
  %expanded_4 = tensor.expand_shape %10 [[0, 1], [2], [3]] output_shape [1, 32, 2, 64] : tensor<32x2x64xf16> into tensor<1x32x2x64xf16>
  %cst = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]]]> : tensor<1x1x1x64xi32>
  %11 = "tosa.const"() <{values = dense<1> : tensor<1x32x2x64xi32>}> : () -> tensor<1x32x2x64xi32>
  %12 = tosa.mul %cst, %11, %shift : (tensor<1x1x1x64xi32>, tensor<1x32x2x64xi32>, tensor<1xi8>) -> tensor<1x32x2x64xi32>
  %13 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %14 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %fused = tosa.mul %expanded_4, %14, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %cst_6 = arith.constant dense<[[[[0], [1]]]]> : tensor<1x1x2x1xi32>
  %15 = tosa.mul %cst_6, %11, %shift : (tensor<1x1x2x1xi32>, tensor<1x32x2x64xi32>, tensor<1xi8>) -> tensor<1x32x2x64xi32>
  %16 = tosa.greater %12, %15 : (tensor<1x32x2x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi1>
  %17 = tosa.cast %16 : (tensor<1x32x2x64xi1>) -> tensor<1x32x2x64xi32>
  %18 = tosa.cast %17 : (tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi8>
  %19 = tosa.cast %18 : (tensor<1x32x2x64xi8>) -> tensor<1x32x2x64xi1>
  %20 = tosa.select %19, %13, %fused : (tensor<1x32x2x64xi1>, tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %expanded_7 = tensor.expand_shape %3 [[0], [1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<1x32xi32> into tensor<1x32x1x1xi32>
  %21 = tosa.mul %expanded_7, %11, %shift : (tensor<1x32x1x1xi32>, tensor<1x32x2x64xi32>, tensor<1xi8>) -> tensor<1x32x2x64xi32>
  %22 = tosa.greater %12, %21 : (tensor<1x32x2x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi1>
  %23 = tosa.cast %22 : (tensor<1x32x2x64xi1>) -> tensor<1x32x2x64xi32>
  %24 = tosa.cast %23 : (tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi8>
  %26 = tosa.cast %24 : (tensor<1x32x2x64xi8>) -> tensor<1x32x2x64xi1>
  %27 = tosa.select %26, %13, %20 : (tensor<1x32x2x64xi1>, tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %28 = tosa.reduce_max %27 {axis = 3 : i32} : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x1xf16>
  %29 = tosa.sub %27, %28 : (tensor<1x32x2x64xf16>, tensor<1x32x2x1xf16>) -> tensor<1x32x2x64xf16>
  %30 = tosa.exp %29 : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %31 = tosa.reduce_sum %30 {axis = 3 : i32} : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x1xf16>
  %32 = tosa.reciprocal %31 : (tensor<1x32x2x1xf16>) -> tensor<1x32x2x1xf16>
  %33 = tosa.mul %30, %32, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x1xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %collapsed_8 = tensor.collapse_shape %33 [[0, 1], [2], [3]] : tensor<1x32x2x64xf16> into tensor<32x2x64xf16>
  %expanded_9 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [32, 64, 128] : tensor<262144xf16> into tensor<32x64x128xf16>
  %34 = tosa.matmul %collapsed_8, %expanded_9, %a_zp, %b_zp : (tensor<32x2x64xf16>, tensor<32x64x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x2x128xf16>
  %expanded_10 = tensor.expand_shape %34 [[0, 1], [2], [3]] output_shape [1, 32, 2, 128] : tensor<32x2x128xf16> into tensor<1x32x2x128xf16>
  %35 = tosa.transpose %expanded_10 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x32x2x128xf16>) -> tensor<1x2x32x128xf16>
  %collapsed_11 = tensor.collapse_shape %35 [[0], [1], [2, 3]] : tensor<1x2x32x128xf16> into tensor<1x2x4096xf16>
  %collapsed_12 = tensor.collapse_shape %35 [[0, 1, 2, 3]] : tensor<1x2x32x128xf16> into tensor<8192xf16>
  return %collapsed_12 : tensor<8192xf16>
}

// CHECK-LABEL: func @mlir_causal_attention2
// CHECK: rock.attention
// CHECK: currentSeqLen = (%{{.*}} : tensor<32xi32>)
// CHECK: causal
func.func @mlir_causal_attention2(%arg0: tensor<24576xf16>, %arg1: tensor<262144xf16>, %arg2: tensor<262144xf16>, %arg3: tensor<1xi32>) -> tensor<8192xf16> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %cst = arith.constant dense<[[[[0], [1]]]]> : tensor<1x1x2x1xi32>
  %0 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %1 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 = "tosa.const"() <{values = dense<1> : tensor<1x32x2x64xi32>}> : () -> tensor<1x32x2x64xi32>
  %cst_0 = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]]]> : tensor<1x1x1x64xi32>
  %5 = "tosa.const"() <{values = dense<1> : tensor<1x32xi32>}> : () -> tensor<1x32xi32>
  %expanded = tensor.expand_shape %arg3 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
  %6 = tosa.mul %expanded, %5, %shift : (tensor<1x1xi32>, tensor<1x32xi32>, tensor<1xi8>) -> tensor<1x32xi32>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 32, 64, 128] : tensor<262144xf16> into tensor<1x32x64x128xf16>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 2, 96, 128] : tensor<24576xf16> into tensor<1x2x96x128xf16>
  %7 = tosa.transpose %expanded_2 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x96x128xf16>) -> tensor<1x96x2x128xf16>
  %extracted_slice = tensor.extract_slice %7[0, 0, 0, 0] [1, 32, 2, 128] [1, 1, 1, 1] : tensor<1x96x2x128xf16> to tensor<1x32x2x128xf16>
  %8 = tosa.transpose %expanded_1 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x32x64x128xf16>) -> tensor<1x32x128x64xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x2x128xf16> into tensor<32x2x128xf16>
  %collapsed_3 = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x32x128x64xf16> into tensor<32x128x64xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %9 = tosa.matmul %collapsed, %collapsed_3, %a_zp, %b_zp : (tensor<32x2x128xf16>, tensor<32x128x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x2x64xf16>
  %expanded_4 = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [1, 32, 2, 64] : tensor<32x2x64xf16> into tensor<1x32x2x64xf16>
  %10 = tosa.mul %cst_0, %2, %shift : (tensor<1x1x1x64xi32>, tensor<1x32x2x64xi32>, tensor<1xi8>) -> tensor<1x32x2x64xi32>
  %11 = tosa.mul %cst, %2, %shift : (tensor<1x1x2x1xi32>, tensor<1x32x2x64xi32>, tensor<1xi8>) -> tensor<1x32x2x64xi32>
  %fused = tosa.mul %expanded_4, %0, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %12 = tosa.greater %10, %11 : (tensor<1x32x2x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi1>
  %13 = tosa.cast %12 : (tensor<1x32x2x64xi1>) -> tensor<1x32x2x64xi32>
  %14 = tosa.cast %13 : (tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi8>
  %15 = tosa.cast %14 : (tensor<1x32x2x64xi8>) -> tensor<1x32x2x64xi1>
  %16 = tosa.select %15, %1, %fused : (tensor<1x32x2x64xi1>, tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %expanded_5 = tensor.expand_shape %6 [[0], [1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<1x32xi32> into tensor<1x32x1x1xi32>
  %17 = tosa.mul %expanded_5, %2, %shift : (tensor<1x32x1x1xi32>, tensor<1x32x2x64xi32>, tensor<1xi8>) -> tensor<1x32x2x64xi32>
  %18 = tosa.greater %10, %17 : (tensor<1x32x2x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi1>
  %19 = tosa.cast %18 : (tensor<1x32x2x64xi1>) -> tensor<1x32x2x64xi32>
  %20 = tosa.cast %19 : (tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi8>
  %22 = tosa.cast %20 : (tensor<1x32x2x64xi8>) -> tensor<1x32x2x64xi1>
  %23 = tosa.select %22, %1, %16 : (tensor<1x32x2x64xi1>, tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %24 = tosa.reduce_max %23 {axis = 3 : i32} : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x1xf16>
  %25 = tosa.sub %23, %24 : (tensor<1x32x2x64xf16>, tensor<1x32x2x1xf16>) -> tensor<1x32x2x64xf16>
  %26 = tosa.exp %25 : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %27 = tosa.reduce_sum %26 {axis = 3 : i32} : (tensor<1x32x2x64xf16>) -> tensor<1x32x2x1xf16>
  %28 = tosa.reciprocal %27 : (tensor<1x32x2x1xf16>) -> tensor<1x32x2x1xf16>
  %29 = tosa.mul %26, %28, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x1xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %collapsed_6 = tensor.collapse_shape %29 [[0, 1], [2], [3]] : tensor<1x32x2x64xf16> into tensor<32x2x64xf16>
  %expanded_7 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [32, 64, 128] : tensor<262144xf16> into tensor<32x64x128xf16>
  %30 = tosa.matmul %collapsed_6, %expanded_7, %a_zp, %b_zp : (tensor<32x2x64xf16>, tensor<32x64x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x2x128xf16>
  %expanded_8 = tensor.expand_shape %30 [[0, 1], [2], [3]] output_shape [1, 32, 2, 128] : tensor<32x2x128xf16> into tensor<1x32x2x128xf16>
  %31 = tosa.transpose %expanded_8 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x32x2x128xf16>) -> tensor<1x2x32x128xf16>
  %collapsed_9 = tensor.collapse_shape %31 [[0, 1, 2, 3]] : tensor<1x2x32x128xf16> into tensor<8192xf16>
  return %collapsed_9 : tensor<8192xf16>
}

// CHECK-LABEL: func @mlir_causal_attention3
// CHECK: rock.attention
// CHECK: currentSeqLen = (%{{.*}} : tensor<32xi32>)
// CHECK: causal
func.func @mlir_causal_attention3(%arg0: tensor<1xi32>, %arg1: tensor<49152xf16>, %arg2: tensor<1048576xf16>, %arg3: tensor<1048576xf16>) -> tensor<16384xf16> attributes {arch = "gfx942", kernel} {
  %0 = "tosa.const"() <{values = dense<"0x00010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101000001010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010000000101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010100000000010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101"> : tensor<1x1x4x256xi8>}> : () -> tensor<1x1x4x256xi8>
  %1 = "tosa.const"() <{values = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000"> : tensor<1x1x1x256xi32>}> : () -> tensor<1x1x1x256xi32>
  %2 = tosa.const_shape  {values = dense<16384> : tensor<1xindex>} : () -> !tosa.shape<1>
  %3 = tosa.const_shape  {values = dense<[32, 256, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<[32, 4, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %5 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x32x4x256xf16>}> : () -> tensor<1x32x4x256xf16>
  %6 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %7 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x4x256xf16>}> : () -> tensor<1x32x4x256xf16>
  %8 = tosa.const_shape  {values = dense<[1, 32, 4, 256]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %9 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %10 = tosa.const_shape  {values = dense<[32, 128, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %11 = tosa.const_shape  {values = dense<[32, 4, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %12 = tosa.const_shape  {values = dense<[1, 32, 4, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %13 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
  %14 = "tosa.const"() <{values = dense<0> : tensor<1x32x4x256xi8>}> : () -> tensor<1x32x4x256xi8>
  %15 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x4x256xf16>}> : () -> tensor<1x32x4x256xf16>
  %16 = "tosa.const"() <{values = dense<0> : tensor<1x1x1x256xi32>}> : () -> tensor<1x1x1x256xi32>
  %17 = tosa.const_shape  {values = dense<[1, 4, 96, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %18 = tosa.const_shape  {values = dense<[1, 32, 256, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 32, 256, 128] : tensor<1048576xf16> into tensor<1x32x256x128xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 4, 96, 128] : tensor<49152xf16> into tensor<1x4x96x128xf16>
  %19 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x4x96x128xf16>) -> tensor<1x96x4x128xf16>
  %20 = tosa.add %0, %14 : (tensor<1x1x4x256xi8>, tensor<1x32x4x256xi8>) -> tensor<1x32x4x256xi8>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %21 = tosa.add %expanded_1, %16 : (tensor<1x1x1x1xi32>, tensor<1x1x1x256xi32>) -> tensor<1x1x1x256xi32>
  %22 = tosa.greater %1, %21 : (tensor<1x1x1x256xi32>, tensor<1x1x1x256xi32>) -> tensor<1x1x1x256xi1>
  %23 = tosa.cast %22 : (tensor<1x1x1x256xi1>) -> tensor<1x1x1x256xi32>
  %24 = tosa.cast %23 : (tensor<1x1x1x256xi32>) -> tensor<1x1x1x256xi8>
  %25 = tosa.add %24, %14 : (tensor<1x1x1x256xi8>, tensor<1x32x4x256xi8>) -> tensor<1x32x4x256xi8>
  %extracted_slice = tensor.extract_slice %19[0, 0, 0, 0] [1, 32, 4, 128] [1, 1, 1, 1] : tensor<1x96x4x128xf16> to tensor<1x32x4x128xf16>
  %26 = tosa.transpose %expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x32x256x128xf16>) -> tensor<1x32x128x256xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x4x128xf16> into tensor<32x4x128xf16>
  %collapsed_2 = tensor.collapse_shape %26 [[0, 1], [2], [3]] : tensor<1x32x128x256xf16> into tensor<32x128x256xf16>
  %27 = tosa.matmul %collapsed, %collapsed_2, %9, %9 {acc_type = f32} : (tensor<32x4x128xf16>, tensor<32x128x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x4x256xf16>
  %expanded_3 = tensor.expand_shape %27 [[0, 1], [2], [3]] output_shape [1, 32, 4, 256] : tensor<32x4x256xf16> into tensor<1x32x4x256xf16>
  %28 = tosa.mul %expanded_3, %15, %6 : (tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>, tensor<1xi8>) -> tensor<1x32x4x256xf16>
  %29 = tosa.cast %20 : (tensor<1x32x4x256xi8>) -> tensor<1x32x4x256xi1>
  %30 = tosa.select %29, %7, %28 : (tensor<1x32x4x256xi1>, tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %31 = tosa.cast %25 : (tensor<1x32x4x256xi8>) -> tensor<1x32x4x256xi1>
  %32 = tosa.select %31, %7, %30 : (tensor<1x32x4x256xi1>, tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %33 = tosa.reduce_max %32 {axis = 3 : i32} : (tensor<1x32x4x256xf16>) -> tensor<1x32x4x1xf16>
  %34 = tosa.add %33, %5 : (tensor<1x32x4x1xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %35 = tosa.sub %32, %34 : (tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %36 = tosa.exp %35 : (tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %37 = tosa.reduce_sum %36 {axis = 3 : i32} : (tensor<1x32x4x256xf16>) -> tensor<1x32x4x1xf16>
  %38 = tosa.add %37, %5 : (tensor<1x32x4x1xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %39 = tosa.reciprocal %38 : (tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %40 = tosa.mul %36, %39, %6 : (tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>, tensor<1xi8>) -> tensor<1x32x4x256xf16>
  %collapsed_4 = tensor.collapse_shape %40 [[0, 1], [2], [3]] : tensor<1x32x4x256xf16> into tensor<32x4x256xf16>
  %expanded_5 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [32, 256, 128] : tensor<1048576xf16> into tensor<32x256x128xf16>
  %41 = tosa.matmul %collapsed_4, %expanded_5, %9, %9 {acc_type = f32} : (tensor<32x4x256xf16>, tensor<32x256x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x4x128xf16>
  %expanded_6 = tensor.expand_shape %41 [[0, 1], [2], [3]] output_shape [1, 32, 4, 128] : tensor<32x4x128xf16> into tensor<1x32x4x128xf16>
  %42 = tosa.transpose %expanded_6 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x32x4x128xf16>) -> tensor<1x4x32x128xf16>
  %collapsed_7 = tensor.collapse_shape %42 [[0, 1, 2, 3]] : tensor<1x4x32x128xf16> into tensor<16384xf16>
  return %collapsed_7 : tensor<16384xf16>
}

// CHECK-LABEL: func @mlir_causal_attention4
// CHECK: rock.attention
// CHECK: currentSeqLen = (%{{.*}} : tensor<32xi32>)
// CHECK: causal
func.func @mlir_causal_attention4(%arg0: tensor<49152xf16>, %arg1: tensor<1048576xf16>, %arg2: tensor<1xi32>, %arg3: tensor<1048576xf16>) -> tensor<16384xf16> attributes {arch = "gfx942", kernel} {
  %0 = "tosa.const"() <{values = dense<"0x00010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101000001010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010000000101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010100000000010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101"> : tensor<1x1x4x256xi8>}> : () -> tensor<1x1x4x256xi8>
  %1 = "tosa.const"() <{values = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000"> : tensor<1x1x1x256xi32>}> : () -> tensor<1x1x1x256xi32>
  %2 = tosa.const_shape  {values = dense<16384> : tensor<1xindex>} : () -> !tosa.shape<1>
  %3 = tosa.const_shape  {values = dense<[32, 256, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<[32, 4, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %5 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x32x4x256xf16>}> : () -> tensor<1x32x4x256xf16>
  %6 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
  %7 = "tosa.const"() <{values = dense<0> : tensor<1x32x4x256xi8>}> : () -> tensor<1x32x4x256xi8>
  %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %9 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x4x256xf16>}> : () -> tensor<1x32x4x256xf16>
  %10 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x4x256xf16>}> : () -> tensor<1x32x4x256xf16>
  %11 = tosa.const_shape  {values = dense<[1, 32, 4, 256]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %12 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %13 = tosa.const_shape  {values = dense<[32, 128, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %14 = tosa.const_shape  {values = dense<[32, 4, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %15 = tosa.const_shape  {values = dense<[1, 32, 4, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %16 = "tosa.const"() <{values = dense<0> : tensor<1x1x1x256xi32>}> : () -> tensor<1x1x1x256xi32>
  %17 = tosa.const_shape  {values = dense<[1, 4, 96, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %18 = tosa.const_shape  {values = dense<[1, 32, 256, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 32, 256, 128] : tensor<1048576xf16> into tensor<1x32x256x128xf16>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 4, 96, 128] : tensor<49152xf16> into tensor<1x4x96x128xf16>
  %19 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x4x96x128xf16>) -> tensor<1x96x4x128xf16>
  %extracted_slice = tensor.extract_slice %19[0, 0, 0, 0] [1, 32, 4, 128] [1, 1, 1, 1] : tensor<1x96x4x128xf16> to tensor<1x32x4x128xf16>
  %20 = tosa.transpose %expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x32x256x128xf16>) -> tensor<1x32x128x256xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x4x128xf16> into tensor<32x4x128xf16>
  %collapsed_1 = tensor.collapse_shape %20 [[0, 1], [2], [3]] : tensor<1x32x128x256xf16> into tensor<32x128x256xf16>
  %21 = tosa.matmul %collapsed, %collapsed_1, %12, %12 {acc_type = f32} : (tensor<32x4x128xf16>, tensor<32x128x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x4x256xf16>
  %expanded_2 = tensor.expand_shape %21 [[0, 1], [2], [3]] output_shape [1, 32, 4, 256] : tensor<32x4x256xf16> into tensor<1x32x4x256xf16>
  %22 = tosa.mul %expanded_2, %9, %8 : (tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>, tensor<1xi8>) -> tensor<1x32x4x256xf16>
  %23 = tosa.add %0, %7 : (tensor<1x1x4x256xi8>, tensor<1x32x4x256xi8>) -> tensor<1x32x4x256xi8>
  %24 = tosa.cast %23 : (tensor<1x32x4x256xi8>) -> tensor<1x32x4x256xi1>
  %25 = tosa.select %24, %10, %22 : (tensor<1x32x4x256xi1>, tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %expanded_3 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %26 = tosa.add %expanded_3, %16 : (tensor<1x1x1x1xi32>, tensor<1x1x1x256xi32>) -> tensor<1x1x1x256xi32>
  %27 = tosa.greater %1, %26 : (tensor<1x1x1x256xi32>, tensor<1x1x1x256xi32>) -> tensor<1x1x1x256xi1>
  %28 = tosa.cast %27 : (tensor<1x1x1x256xi1>) -> tensor<1x1x1x256xi32>
  %29 = tosa.cast %28 : (tensor<1x1x1x256xi32>) -> tensor<1x1x1x256xi8>
  %30 = tosa.add %29, %7 : (tensor<1x1x1x256xi8>, tensor<1x32x4x256xi8>) -> tensor<1x32x4x256xi8>
  %31 = tosa.cast %30 : (tensor<1x32x4x256xi8>) -> tensor<1x32x4x256xi1>
  %32 = tosa.select %31, %10, %25 : (tensor<1x32x4x256xi1>, tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %33 = tosa.reduce_max %32 {axis = 3 : i32} : (tensor<1x32x4x256xf16>) -> tensor<1x32x4x1xf16>
  %34 = tosa.add %33, %5 : (tensor<1x32x4x1xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %35 = tosa.sub %32, %34 : (tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %36 = tosa.exp %35 : (tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %37 = tosa.reduce_sum %36 {axis = 3 : i32} : (tensor<1x32x4x256xf16>) -> tensor<1x32x4x1xf16>
  %38 = tosa.add %37, %5 : (tensor<1x32x4x1xf16>, tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %39 = tosa.reciprocal %38 : (tensor<1x32x4x256xf16>) -> tensor<1x32x4x256xf16>
  %40 = tosa.mul %36, %39, %8 : (tensor<1x32x4x256xf16>, tensor<1x32x4x256xf16>, tensor<1xi8>) -> tensor<1x32x4x256xf16>
  %collapsed_4 = tensor.collapse_shape %40 [[0, 1], [2], [3]] : tensor<1x32x4x256xf16> into tensor<32x4x256xf16>
  %expanded_5 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [32, 256, 128] : tensor<1048576xf16> into tensor<32x256x128xf16>
  %41 = tosa.matmul %collapsed_4, %expanded_5, %12, %12 {acc_type = f32} : (tensor<32x4x256xf16>, tensor<32x256x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x4x128xf16>
  %expanded_6 = tensor.expand_shape %41 [[0, 1], [2], [3]] output_shape [1, 32, 4, 128] : tensor<32x4x128xf16> into tensor<1x32x4x128xf16>
  %42 = tosa.transpose %expanded_6 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x32x4x128xf16>) -> tensor<1x4x32x128xf16>
  %collapsed_7 = tensor.collapse_shape %42 [[0, 1, 2, 3]] : tensor<1x4x32x128xf16> into tensor<16384xf16>
  return %collapsed_7 : tensor<16384xf16>
}

