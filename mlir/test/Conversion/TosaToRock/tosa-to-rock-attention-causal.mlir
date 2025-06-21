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
  %11 = "tosa.const"() <{values = dense<0> : tensor<1x32x2x64xi32>}> : () -> tensor<1x32x2x64xi32>
  %12 = tosa.add %cst, %11 : (tensor<1x1x1x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi32>
  %13 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %14 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %fused = tosa.mul %expanded_4, %14, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %cst_6 = arith.constant dense<[[[[0], [1]]]]> : tensor<1x1x2x1xi32>
  %15 = tosa.add %cst_6, %11 : (tensor<1x1x2x1xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi32>
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
  %2 = "tosa.const"() <{values = dense<0> : tensor<1x32xi32>}> : () -> tensor<1x32xi32>
  %3 = tosa.add %1, %2 : (tensor<1x1xi32>, tensor<1x32xi32>) -> tensor<1x32xi32>
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
  %11 = "tosa.const"() <{values = dense<0> : tensor<1x32x2x64xi32>}> : () -> tensor<1x32x2x64xi32>
  %12 = tosa.add %cst, %11 : (tensor<1x1x1x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi32>
  %13 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %14 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x2x64xf16>}> : () -> tensor<1x32x2x64xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %fused = tosa.mul %expanded_4, %14, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %cst_6 = arith.constant dense<[[[[0], [1]]]]> : tensor<1x1x2x1xi32>
  %15 = tosa.add %cst_6, %11 : (tensor<1x1x2x1xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi32>
  %16 = tosa.greater %12, %15 : (tensor<1x32x2x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi1>
  %17 = tosa.cast %16 : (tensor<1x32x2x64xi1>) -> tensor<1x32x2x64xi32>
  %18 = tosa.cast %17 : (tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi8>
  %19 = tosa.cast %18 : (tensor<1x32x2x64xi8>) -> tensor<1x32x2x64xi1>
  %20 = tosa.select %19, %13, %fused : (tensor<1x32x2x64xi1>, tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %expanded_7 = tensor.expand_shape %3 [[0], [1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<1x32xi32> into tensor<1x32x1x1xi32>
  %21 = tosa.add %expanded_7, %11 : (tensor<1x32x1x1xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi32>
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
  %2 = "tosa.const"() <{values = dense<0> : tensor<1x32x2x64xi32>}> : () -> tensor<1x32x2x64xi32>
  %cst_0 = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]]]> : tensor<1x1x1x64xi32>
  %5 = "tosa.const"() <{values = dense<0> : tensor<1x32xi32>}> : () -> tensor<1x32xi32>
  %expanded = tensor.expand_shape %arg3 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
  %6 = tosa.add %expanded, %5 : (tensor<1x1xi32>, tensor<1x32xi32>) -> tensor<1x32xi32>
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
  %10 = tosa.add %cst_0, %2 : (tensor<1x1x1x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi32>
  %11 = tosa.add %cst, %2 : (tensor<1x1x2x1xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %fused = tosa.mul %expanded_4, %0, %shift : (tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>, tensor<1xi8>) -> tensor<1x32x2x64xf16>
  %12 = tosa.greater %10, %11 : (tensor<1x32x2x64xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi1>
  %13 = tosa.cast %12 : (tensor<1x32x2x64xi1>) -> tensor<1x32x2x64xi32>
  %14 = tosa.cast %13 : (tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi8>
  %15 = tosa.cast %14 : (tensor<1x32x2x64xi8>) -> tensor<1x32x2x64xi1>
  %16 = tosa.select %15, %1, %fused : (tensor<1x32x2x64xi1>, tensor<1x32x2x64xf16>, tensor<1x32x2x64xf16>) -> tensor<1x32x2x64xf16>
  %expanded_5 = tensor.expand_shape %6 [[0], [1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<1x32xi32> into tensor<1x32x1x1xi32>
  %17 = tosa.add %expanded_5, %2 : (tensor<1x32x1x1xi32>, tensor<1x32x2x64xi32>) -> tensor<1x32x2x64xi32>
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
