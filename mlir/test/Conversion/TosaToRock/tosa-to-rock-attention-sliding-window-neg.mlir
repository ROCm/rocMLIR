// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -split-input-file -verify-diagnostics -o -| FileCheck %s

// A sliding-window mask without a separate KV-cache mask must adopt its
// validated seq-len operand as currentSeqLen and preserve its clip.
// CHECK-LABEL: func @sliding_window_no_kvcache
// CHECK: %[[MAX:.*]] = tosa.maximum
// CHECK: %[[CLIP:.*]] = tosa.minimum %[[MAX]]
// CHECK: rock.attention
// CHECK: currentSeqLen = (%[[CLIP]]
// CHECK: slidingWindowSize = 3
func.func @sliding_window_no_kvcache(%arg0: tensor<1xi32>, %arg1: tensor<12xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<4> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %4 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x1x8xf32>}> : () -> tensor<1x2x1x8xf32>
  %5 = "tosa.const"() <{values = dense<1> : tensor<1x2x1x8xi8>}> : () -> tensor<1x2x1x8xi8>
  %7 = "tosa.const"() <{values = dense<1> : tensor<8x1x1x1xi32>}> : () -> tensor<8x1x1x1xi32>
  %8 = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %9 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %11 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %16 = "tosa.const"() <{values = dense<-3> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %17 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %20 = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>}> : () -> tensor<8xi32>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 2, 8, 2] : tensor<32xf16> into tensor<1x2x8x2xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 6, 1, 2] : tensor<12xf16> into tensor<1x6x1x2xf16>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %23 = tosa.maximum %expanded_1, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %24 = tosa.minimum %23, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %extracted_slice = tensor.extract_slice %expanded_0[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %25 = tosa.transpose %expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %collapsed_2 = tensor.collapse_shape %25 [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %26 = tosa.matmul %collapsed, %collapsed_2, %11, %11 {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %expanded_3 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %27 = tosa.mul %expanded_3, %8, %17 : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %28 = tosa.add %24, %16 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %29 = tosa.mul %28, %7, %17 : (tensor<1x1x1x1xi32>, tensor<8x1x1x1xi32>, tensor<1xi8>) -> tensor<8x1x1x1xi32>
  %collapsed_4 = tensor.collapse_shape %29 [[0, 1, 2, 3]] : tensor<8x1x1x1xi32> into tensor<8xi32>
  %30 = tosa.greater %collapsed_4, %20 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  %31 = tosa.cast %30 : (tensor<8xi1>) -> tensor<8xi32>
  %32 = tosa.cast %31 : (tensor<8xi32>) -> tensor<8xi8>
  %expanded_5 = tensor.expand_shape %32 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %33 = tosa.mul %expanded_5, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %34 = tosa.cast %33 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %35 = tosa.select %34, %9, %27 : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %43 = tosa.cast %35 : (tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf32>
  %44 = tosa.reduce_max %43 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %45 = tosa.mul %44, %4, %17 : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %46 = tosa.sub %43, %45 : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %47 = tosa.exp %46 : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %48 = tosa.reduce_sum %47 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %49 = tosa.mul %48, %4, %17 : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %50 = tosa.reciprocal %49 : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %51 = tosa.mul %47, %50, %17 : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %52 = tosa.cast %51 : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf16>
  %collapsed_6 = tensor.collapse_shape %52 [[0, 1], [2], [3]] : tensor<1x2x1x8xf16> into tensor<2x1x8xf16>
  %expanded_7 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [2, 8, 2] : tensor<32xf16> into tensor<2x8x2xf16>
  %53 = tosa.matmul %collapsed_6, %expanded_7, %11, %11 {acc_type = f32} : (tensor<2x1x8xf16>, tensor<2x8x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x2xf16>
  %expanded_8 = tensor.expand_shape %53 [[0, 1], [2], [3]] output_shape [1, 2, 1, 2] : tensor<2x1x2xf16> into tensor<1x2x1x2xf16>
  %54 = tosa.transpose %expanded_8 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x1x2xf16>) -> tensor<1x1x2x2xf16>
  %collapsed_9 = tensor.collapse_shape %54 [[0, 1, 2, 3]] : tensor<1x1x2x2xf16> into tensor<4xf16>
  return %collapsed_9 : tensor<4xf16>
}

// -----

// A greater(x - window, col) mask whose x is not currentSeqLen must not be
// classified as sliding-window attention.
// CHECK-LABEL: func @not_sliding_window_wrong_operand
// CHECK: rock.attention
// CHECK: currentSeqLen =
// CHECK-NOT: slidingWindowSize
func.func @not_sliding_window_wrong_operand(%arg0: tensor<1xi32>, %arg1: tensor<12xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<4> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %4 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x1x8xf32>}> : () -> tensor<1x2x1x8xf32>
  %5 = "tosa.const"() <{values = dense<1> : tensor<1x2x1x8xi8>}> : () -> tensor<1x2x1x8xi8>
  %7 = "tosa.const"() <{values = dense<1> : tensor<8x1x1x1xi32>}> : () -> tensor<8x1x1x1xi32>
  %8 = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %9 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %11 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %16 = "tosa.const"() <{values = dense<-3> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %17 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %18 = "tosa.const"() <{values = dense<1> : tensor<1x1x1x8xi32>}> : () -> tensor<1x1x1x8xi32>
  %20 = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>}> : () -> tensor<8xi32>
  %cst = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 2, 8, 2] : tensor<32xf16> into tensor<1x2x8x2xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 6, 1, 2] : tensor<12xf16> into tensor<1x6x1x2xf16>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %23 = tosa.maximum %expanded_1, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %24 = tosa.minimum %23, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %extracted_slice = tensor.extract_slice %expanded_0[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %25 = tosa.transpose %expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %collapsed_2 = tensor.collapse_shape %25 [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %26 = tosa.matmul %collapsed, %collapsed_2, %11, %11 {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %expanded_3 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %27 = tosa.mul %expanded_3, %8, %17 : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  // Subtract the window size from a constant, not from currentSeqLen.
  %28 = tosa.add %0, %16 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %29 = tosa.mul %28, %7, %17 : (tensor<1x1x1x1xi32>, tensor<8x1x1x1xi32>, tensor<1xi8>) -> tensor<8x1x1x1xi32>
  %collapsed_4 = tensor.collapse_shape %29 [[0, 1, 2, 3]] : tensor<8x1x1x1xi32> into tensor<8xi32>
  %30 = tosa.greater %collapsed_4, %20 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  %31 = tosa.cast %30 : (tensor<8xi1>) -> tensor<8xi32>
  %32 = tosa.cast %31 : (tensor<8xi32>) -> tensor<8xi8>
  %expanded_5 = tensor.expand_shape %32 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %33 = tosa.mul %expanded_5, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %34 = tosa.cast %33 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %35 = tosa.select %34, %9, %27 : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %36 = tosa.mul %24, %18, %17 : (tensor<1x1x1x1xi32>, tensor<1x1x1x8xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
  %37 = tosa.greater %cst, %36 : (tensor<1x1x1x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi1>
  %38 = tosa.cast %37 : (tensor<1x1x1x8xi1>) -> tensor<1x1x1x8xi32>
  %39 = tosa.cast %38 : (tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi8>
  %40 = tosa.mul %39, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %41 = tosa.cast %40 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %42 = tosa.select %41, %9, %35 : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %43 = tosa.cast %42 : (tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf32>
  %44 = tosa.reduce_max %43 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %45 = tosa.mul %44, %4, %17 : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %46 = tosa.sub %43, %45 : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %47 = tosa.exp %46 : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %48 = tosa.reduce_sum %47 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %49 = tosa.mul %48, %4, %17 : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %50 = tosa.reciprocal %49 : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %51 = tosa.mul %47, %50, %17 : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %52 = tosa.cast %51 : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf16>
  %collapsed_6 = tensor.collapse_shape %52 [[0, 1], [2], [3]] : tensor<1x2x1x8xf16> into tensor<2x1x8xf16>
  %expanded_7 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [2, 8, 2] : tensor<32xf16> into tensor<2x8x2xf16>
  %53 = tosa.matmul %collapsed_6, %expanded_7, %11, %11 {acc_type = f32} : (tensor<2x1x8xf16>, tensor<2x8x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x2xf16>
  %expanded_8 = tensor.expand_shape %53 [[0, 1], [2], [3]] output_shape [1, 2, 1, 2] : tensor<2x1x2xf16> into tensor<1x2x1x2xf16>
  %54 = tosa.transpose %expanded_8 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x1x2xf16>) -> tensor<1x1x2x2xf16>
  %collapsed_9 = tensor.collapse_shape %54 [[0, 1, 2, 3]] : tensor<1x1x2x2xf16> into tensor<4xf16>
  return %collapsed_9 : tensor<4xf16>
}
