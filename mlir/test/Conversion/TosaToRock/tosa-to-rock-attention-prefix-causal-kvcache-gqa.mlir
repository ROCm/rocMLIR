// RUN: rocmlir-opt --tosa-to-rock %s | FileCheck %s

// This test verifies that TosaToRock correctly detects the combination of:
// - GQA (Grouped Query Attention)
// - KVCache
// - Prefix Causal Attention

// CHECK-LABEL: func @attention_gqa_kvcache_prefix_causal
// CHECK: rock.attention
// CHECK-NEXT: qk = %{{.*}} * %{{.*}}
// CHECK-NEXT: currentSeqLen = (%{{.*}}
// CHECK-DAG: prefixOffset = (%{{.*}}
// CHECK-DAG: causal

module {
  func.func @attention_gqa_kvcache_prefix_causal(%arg0: tensor<1xi32> {mhal.read_access}, %arg1: tensor<9216xf16> {mhal.read_access}, %arg2: tensor<2048xf16> {mhal.read_access}, %arg3: tensor<2048xf16> {mhal.read_access}, %arg4: tensor<2xi32> {mhal.read_access}) -> (tensor<7168xf16> {mhal.write_access}) attributes {kernel} {
    %0 = "tosa.const"() <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
    %1 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x14x4x8xf32>}> : () -> tensor<2x14x4x8xf32>
    %2 = "tosa.const"() <{values = dense<0xFC00> : tensor<2x14x4x8xf16>}> : () -> tensor<2x14x4x8xf16>
    %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x64x8xf16>}> : () -> tensor<2x2x7x64x8xf16>
    %5 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x7x8x64xf16>}> : () -> tensor<2x2x7x8x64xf16>
    %6 = "tosa.const"() <{values = dense<1> : tensor<2x14x4x8xi8>}> : () -> tensor<2x14x4x8xi8>
    %7 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<2x14x4x8xf16>}> : () -> tensor<2x14x4x8xf16>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = "tosa.const"() <{values = dense<1> : tensor<4x8xi32>}> : () -> tensor<4x8xi32>
    %10 = "tosa.const"() <{values = dense<1> : tensor<4x1xi32>}> : () -> tensor<4x1xi32>
    %11 = "tosa.const"() <{values = dense<1> : tensor<2x8xi32>}> : () -> tensor<2x8xi32>
    %12 = "tosa.const"() <{values = dense<[[0], [1], [2], [3]]> : tensor<4x1xi32>}> : () -> tensor<4x1xi32>
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 4, 18, 64] : tensor<9216xf16> into tensor<2x4x18x64xf16>
    %13 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x4x18x64xf16>) -> tensor<2x18x4x64xf16>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
    %14 = tosa.mul %expanded_0, %10, %8 : (tensor<1x1xi32>, tensor<4x1xi32>, tensor<1xi8>) -> tensor<4x1xi32>
    %15 = tosa.add %14, %12 : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x1xi32>
    %16 = tosa.mul %15, %9, %8 : (tensor<4x1xi32>, tensor<4x8xi32>, tensor<1xi8>) -> tensor<4x8xi32>
    %17 = tosa.mul %0, %9, %8 : (tensor<1x8xi32>, tensor<4x8xi32>, tensor<1xi8>) -> tensor<4x8xi32>
    %18 = tosa.greater %17, %16 : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
    %19 = tosa.cast %18 : (tensor<4x8xi1>) -> tensor<4x8xi32>
    %20 = tosa.cast %19 : (tensor<4x8xi32>) -> tensor<4x8xi8>
    %expanded_1 = tensor.expand_shape %20 [[0, 1, 2], [3]] output_shape [1, 1, 4, 8] : tensor<4x8xi8> into tensor<1x1x4x8xi8>
    %21 = tosa.mul %expanded_1, %6, %8 : (tensor<1x1x4x8xi8>, tensor<2x14x4x8xi8>, tensor<1xi8>) -> tensor<2x14x4x8xi8>
    %22 = tosa.mul %0, %11, %8 : (tensor<1x8xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
    %expanded_2 = tensor.expand_shape %arg4 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
    %23 = tosa.mul %expanded_2, %11, %8 : (tensor<2x1xi32>, tensor<2x8xi32>, tensor<1xi8>) -> tensor<2x8xi32>
    %24 = tosa.greater %22, %23 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
    %25 = tosa.cast %24 : (tensor<2x8xi1>) -> tensor<2x8xi32>
    %26 = tosa.cast %25 : (tensor<2x8xi32>) -> tensor<2x8xi8>
    %expanded_3 = tensor.expand_shape %26 [[0, 1, 2], [3]] output_shape [2, 1, 1, 8] : tensor<2x8xi8> into tensor<2x1x1x8xi8>
    %27 = tosa.mul %expanded_3, %6, %8 : (tensor<2x1x1x8xi8>, tensor<2x14x4x8xi8>, tensor<1xi8>) -> tensor<2x14x4x8xi8>
    %extracted_slice = tensor.extract_slice %13[0, 0, 0, 0] [2, 14, 4, 64] [1, 1, 1, 1] : tensor<2x18x4x64xf16> to tensor<2x14x4x64xf16>
    %expanded_4 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
    %28 = tosa.mul %expanded_4, %5, %8 : (tensor<2x2x1x8x64xf16>, tensor<2x2x7x8x64xf16>, tensor<1xi8>) -> tensor<2x2x7x8x64xf16>
    %expanded_5 = tensor.expand_shape %arg3 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 8, 64] : tensor<2048xf16> into tensor<2x2x1x8x64xf16>
    %29 = tosa.transpose %expanded_5 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<2x2x1x8x64xf16>) -> tensor<2x2x1x64x8xf16>
    %30 = tosa.mul %29, %4, %8 : (tensor<2x2x1x64x8xf16>, tensor<2x2x7x64x8xf16>, tensor<1xi8>) -> tensor<2x2x7x64x8xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<2x14x4x64xf16> into tensor<28x4x64xf16>
    %collapsed_6 = tensor.collapse_shape %30 [[0, 1, 2], [3], [4]] : tensor<2x2x7x64x8xf16> into tensor<28x64x8xf16>
    %31 = tosa.matmul %collapsed, %collapsed_6, %3, %3 {acc_type = f32} : (tensor<28x4x64xf16>, tensor<28x64x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x4x8xf16>
    %expanded_7 = tensor.expand_shape %31 [[0, 1], [2], [3]] output_shape [2, 14, 4, 8] : tensor<28x4x8xf16> into tensor<2x14x4x8xf16>
    %32 = tosa.mul %expanded_7, %7, %8 : (tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>, tensor<1xi8>) -> tensor<2x14x4x8xf16>
    %33 = tosa.cast %21 : (tensor<2x14x4x8xi8>) -> tensor<2x14x4x8xi1>
    %34 = tosa.select %33, %2, %32 : (tensor<2x14x4x8xi1>, tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf16>
    %35 = tosa.cast %27 : (tensor<2x14x4x8xi8>) -> tensor<2x14x4x8xi1>
    %36 = tosa.select %35, %2, %34 : (tensor<2x14x4x8xi1>, tensor<2x14x4x8xf16>, tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf16>
    %37 = tosa.cast %36 : (tensor<2x14x4x8xf16>) -> tensor<2x14x4x8xf32>
    %38 = tosa.reduce_max %37 {axis = 3 : i32} : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x1xf32>
    %39 = tosa.mul %38, %1, %8 : (tensor<2x14x4x1xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
    %40 = tosa.sub %37, %39 : (tensor<2x14x4x8xf32>, tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
    %41 = tosa.exp %40 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
    %42 = tosa.reduce_sum %41 {axis = 3 : i32} : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x1xf32>
    %43 = tosa.mul %42, %1, %8 : (tensor<2x14x4x1xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
    %44 = tosa.reciprocal %43 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf32>
    %45 = tosa.mul %41, %44, %8 : (tensor<2x14x4x8xf32>, tensor<2x14x4x8xf32>, tensor<1xi8>) -> tensor<2x14x4x8xf32>
    %46 = tosa.cast %45 : (tensor<2x14x4x8xf32>) -> tensor<2x14x4x8xf16>
    %collapsed_8 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<2x14x4x8xf16> into tensor<28x4x8xf16>
    %collapsed_9 = tensor.collapse_shape %28 [[0, 1, 2], [3], [4]] : tensor<2x2x7x8x64xf16> into tensor<28x8x64xf16>
    %47 = tosa.matmul %collapsed_8, %collapsed_9, %3, %3 {acc_type = f32} : (tensor<28x4x8xf16>, tensor<28x8x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<28x4x64xf16>
    %expanded_10 = tensor.expand_shape %47 [[0, 1], [2], [3]] output_shape [2, 14, 4, 64] : tensor<28x4x64xf16> into tensor<2x14x4x64xf16>
    %48 = tosa.transpose %expanded_10 {perms = array<i32: 0, 2, 1, 3>} : (tensor<2x14x4x64xf16>) -> tensor<2x4x14x64xf16>
    %collapsed_11 = tensor.collapse_shape %48 [[0, 1, 2, 3]] : tensor<2x4x14x64xf16> into tensor<7168xf16>
    return %collapsed_11 : tensor<7168xf16>
  }
}

