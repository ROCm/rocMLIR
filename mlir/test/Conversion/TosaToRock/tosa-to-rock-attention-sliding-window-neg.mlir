// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -split-input-file -verify-diagnostics -o -| FileCheck %s

// Edge case 1: a sliding-window-shaped lower mask WITHOUT a separate KV-cache
// upper mask. Folding this as sliding-window attention would set currentSeqLen
// and introduce a new upper mask for keys after that position. Keep the lower
// mask in the elementwise region instead.
// CHECK-LABEL: func @sliding_window_no_kvcache
// CHECK: rock.attention
// CHECK-NOT: currentSeqLen
// CHECK-NOT: slidingWindowSize
// CHECK: qk = elementwise
// CHECK: tosa.maximum
// CHECK: tosa.minimum
// CHECK: tosa.greater
// CHECK: tosa.select
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

// The KV-cache and sliding-window masks reference different currentSeqLen
// block arguments. They cannot be represented by one folded attention op, so
// neither mask should be folded.
// CHECK-LABEL: func @sliding_window_kvcache_mismatched_seqlen
// CHECK: rock.attention
// CHECK-NOT: slidingWindowSize
// CHECK-NOT: currentSeqLen
func.func @sliding_window_kvcache_mismatched_seqlen(%arg0: tensor<1xi32>, %arg1: tensor<12xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>, %arg4: tensor<1xi32>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
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
  %expanded_kv = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %23 = tosa.maximum %expanded_1, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %24 = tosa.minimum %23, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %k23 = tosa.maximum %expanded_kv, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %k24 = tosa.minimum %k23, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %extracted_slice = tensor.extract_slice %expanded_0[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %25 = tosa.transpose %expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %collapsed_2 = tensor.collapse_shape %25 [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %26 = tosa.matmul %collapsed, %collapsed_2, %11, %11 {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %expanded_3 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %27 = tosa.mul %expanded_3, %8, %17 : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %k36 = tosa.mul %k24, %18, %17 : (tensor<1x1x1x1xi32>, tensor<1x1x1x8xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
  %k37 = tosa.greater %cst, %k36 : (tensor<1x1x1x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi1>
  %k38 = tosa.cast %k37 : (tensor<1x1x1x8xi1>) -> tensor<1x1x1x8xi32>
  %k39 = tosa.cast %k38 : (tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi8>
  %k40 = tosa.mul %k39, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %k41 = tosa.cast %k40 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %kvsel = tosa.select %k41, %9, %27 : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %28 = tosa.add %24, %16 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %29 = tosa.mul %28, %7, %17 : (tensor<1x1x1x1xi32>, tensor<8x1x1x1xi32>, tensor<1xi8>) -> tensor<8x1x1x1xi32>
  %collapsed_4 = tensor.collapse_shape %29 [[0, 1, 2, 3]] : tensor<8x1x1x1xi32> into tensor<8xi32>
  %30 = tosa.greater %collapsed_4, %20 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  %31 = tosa.cast %30 : (tensor<8xi1>) -> tensor<8xi32>
  %32 = tosa.cast %31 : (tensor<8xi32>) -> tensor<8xi8>
  %expanded_5 = tensor.expand_shape %32 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %33 = tosa.mul %expanded_5, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %34 = tosa.cast %33 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %35 = tosa.select %34, %9, %kvsel : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
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

// The KV-cache and sliding-window masks use the same currentSeqLen block
// argument but different clip bounds. A single folded currentSeqLen cannot
// represent both clamps, so neither mask should be folded.
// CHECK-LABEL: func @sliding_window_kvcache_mismatched_clip
// CHECK: rock.attention
// CHECK-NOT: slidingWindowSize
// CHECK-NOT: currentSeqLen
func.func @sliding_window_kvcache_mismatched_clip(%arg0: tensor<1xi32>, %arg1: tensor<12xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<4> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %2 = "tosa.const"() <{values = dense<2> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
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
  %k23 = tosa.maximum %expanded_1, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %k24 = tosa.minimum %k23, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %23 = tosa.maximum %expanded_1, %2 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %24 = tosa.minimum %23, %2 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %extracted_slice = tensor.extract_slice %expanded_0[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %25 = tosa.transpose %expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %collapsed_2 = tensor.collapse_shape %25 [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %26 = tosa.matmul %collapsed, %collapsed_2, %11, %11 {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %expanded_3 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %27 = tosa.mul %expanded_3, %8, %17 : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %k36 = tosa.mul %k24, %18, %17 : (tensor<1x1x1x1xi32>, tensor<1x1x1x8xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
  %k37 = tosa.greater %cst, %k36 : (tensor<1x1x1x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi1>
  %k38 = tosa.cast %k37 : (tensor<1x1x1x8xi1>) -> tensor<1x1x1x8xi32>
  %k39 = tosa.cast %k38 : (tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi8>
  %k40 = tosa.mul %k39, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %k41 = tosa.cast %k40 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %kvsel = tosa.select %k41, %9, %27 : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %28 = tosa.add %24, %16 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %29 = tosa.mul %28, %7, %17 : (tensor<1x1x1x1xi32>, tensor<8x1x1x1xi32>, tensor<1xi8>) -> tensor<8x1x1x1xi32>
  %collapsed_4 = tensor.collapse_shape %29 [[0, 1, 2, 3]] : tensor<8x1x1x1xi32> into tensor<8xi32>
  %30 = tosa.greater %collapsed_4, %20 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  %31 = tosa.cast %30 : (tensor<8xi1>) -> tensor<8xi32>
  %32 = tosa.cast %31 : (tensor<8xi32>) -> tensor<8xi8>
  %expanded_5 = tensor.expand_shape %32 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %33 = tosa.mul %expanded_5, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %34 = tosa.cast %33 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %35 = tosa.select %34, %9, %kvsel : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
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

// A negative currentSeqLen clip cannot be represented by Rock's unsigned mask
// comparisons. Keep both masks explicit rather than dropping the clamp while
// folding them into rock.attention.
// CHECK-LABEL: func @sliding_window_kvcache_negative_clip
// CHECK: rock.attention
// CHECK-NOT: slidingWindowSize
// CHECK-NOT: currentSeqLen
// CHECK: qk = elementwise
// CHECK: tosa.maximum
// CHECK: tosa.minimum
func.func @sliding_window_kvcache_negative_clip(%arg0: tensor<1xi32>, %arg1: tensor<12xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<4> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %2 = "tosa.const"() <{values = dense<-1> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
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
  %k23 = tosa.maximum %expanded_1, %2 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %k24 = tosa.minimum %k23, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %23 = tosa.maximum %expanded_1, %2 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %24 = tosa.minimum %23, %0 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %extracted_slice = tensor.extract_slice %expanded_0[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %25 = tosa.transpose %expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %collapsed_2 = tensor.collapse_shape %25 [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %26 = tosa.matmul %collapsed, %collapsed_2, %11, %11 {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %expanded_3 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %27 = tosa.mul %expanded_3, %8, %17 : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %k36 = tosa.mul %k24, %18, %17 : (tensor<1x1x1x1xi32>, tensor<1x1x1x8xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
  %k37 = tosa.greater %cst, %k36 : (tensor<1x1x1x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi1>
  %k38 = tosa.cast %k37 : (tensor<1x1x1x8xi1>) -> tensor<1x1x1x8xi32>
  %k39 = tosa.cast %k38 : (tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi8>
  %k40 = tosa.mul %k39, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %k41 = tosa.cast %k40 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %kvsel = tosa.select %k41, %9, %27 : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %28 = tosa.add %24, %16 : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %29 = tosa.mul %28, %7, %17 : (tensor<1x1x1x1xi32>, tensor<8x1x1x1xi32>, tensor<1xi8>) -> tensor<8x1x1x1xi32>
  %collapsed_4 = tensor.collapse_shape %29 [[0, 1, 2, 3]] : tensor<8x1x1x1xi32> into tensor<8xi32>
  %30 = tosa.greater %collapsed_4, %20 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  %31 = tosa.cast %30 : (tensor<8xi1>) -> tensor<8xi32>
  %32 = tosa.cast %31 : (tensor<8xi32>) -> tensor<8xi8>
  %expanded_5 = tensor.expand_shape %32 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %33 = tosa.mul %expanded_5, %5, %17 : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %34 = tosa.cast %33 : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %35 = tosa.select %34, %9, %kvsel : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
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

// A one-sided upper clamp is representable by rematerializing the minimum on
// currentSeqLen. Fold both masks without introducing a lower clamp.
// CHECK-LABEL: func @sliding_window_kvcache_one_sided_clip
// CHECK-NOT: tosa.maximum
// CHECK: %[[CLIP:.*]] = tosa.minimum
// CHECK: rock.attention
// CHECK: currentSeqLen = (%[[CLIP]] : tensor<2xi32>)
// CHECK: slidingWindowSize = 3
func.func @sliding_window_kvcache_one_sided_clip(%arg0: tensor<1xi32>, %arg1: tensor<12xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %clip_max = "tosa.const"() <{values = dense<4> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %softmax_ones = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x1x8xf32>}> : () -> tensor<1x2x1x8xf32>
  %mask_ones = "tosa.const"() <{values = dense<1> : tensor<1x2x1x8xi8>}> : () -> tensor<1x2x1x8xi8>
  %window_broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<8x1x1x1xi32>}> : () -> tensor<8x1x1x1xi32>
  %scale = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %neg_inf = "tosa.const"() <{values = dense<0xFC00> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %zero = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %window_offset = "tosa.const"() <{values = dense<-3> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %kv_broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<1x1x1x8xi32>}> : () -> tensor<1x1x1x8xi32>
  %columns_1d = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>}> : () -> tensor<8xi32>
  %columns_4d = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %keys_expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 2, 8, 2] : tensor<32xf16> into tensor<1x2x8x2xf16>
  %queries_expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 6, 1, 2] : tensor<12xf16> into tensor<1x6x1x2xf16>
  %seq_len = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %clipped_seq_len = tosa.minimum %seq_len, %clip_max : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %queries = tensor.extract_slice %queries_expanded[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %keys = tosa.transpose %keys_expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %queries_collapsed = tensor.collapse_shape %queries [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %keys_collapsed = tensor.collapse_shape %keys [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %scores = tosa.matmul %queries_collapsed, %keys_collapsed, %zero, %zero {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %scores_expanded = tensor.expand_shape %scores [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %scaled_scores = tosa.mul %scores_expanded, %scale, %shift : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %kv_seq_len = tosa.mul %clipped_seq_len, %kv_broadcast_ones, %shift : (tensor<1x1x1x1xi32>, tensor<1x1x1x8xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
  %kv_pred = tosa.greater %columns_4d, %kv_seq_len : (tensor<1x1x1x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi1>
  %kv_pred_i32 = tosa.cast %kv_pred : (tensor<1x1x1x8xi1>) -> tensor<1x1x1x8xi32>
  %kv_pred_i8 = tosa.cast %kv_pred_i32 : (tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi8>
  %kv_pred_broadcast = tosa.mul %kv_pred_i8, %mask_ones, %shift : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %kv_mask = tosa.cast %kv_pred_broadcast : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %kv_masked_scores = tosa.select %kv_mask, %neg_inf, %scaled_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %window_lower = tosa.add %clipped_seq_len, %window_offset : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %window_broadcast = tosa.mul %window_lower, %window_broadcast_ones, %shift : (tensor<1x1x1x1xi32>, tensor<8x1x1x1xi32>, tensor<1xi8>) -> tensor<8x1x1x1xi32>
  %window_flat = tensor.collapse_shape %window_broadcast [[0, 1, 2, 3]] : tensor<8x1x1x1xi32> into tensor<8xi32>
  %window_pred = tosa.greater %window_flat, %columns_1d : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  %window_pred_i32 = tosa.cast %window_pred : (tensor<8xi1>) -> tensor<8xi32>
  %window_pred_i8 = tosa.cast %window_pred_i32 : (tensor<8xi32>) -> tensor<8xi8>
  %window_pred_4d = tensor.expand_shape %window_pred_i8 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %window_pred_broadcast = tosa.mul %window_pred_4d, %mask_ones, %shift : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %window_mask = tosa.cast %window_pred_broadcast : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %masked_scores = tosa.select %window_mask, %neg_inf, %kv_masked_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %scores_f32 = tosa.cast %masked_scores : (tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf32>
  %max = tosa.reduce_max %scores_f32 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %max_broadcast = tosa.mul %max, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %normalized = tosa.sub %scores_f32, %max_broadcast : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %exp = tosa.exp %normalized : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %sum_broadcast = tosa.mul %sum, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %reciprocal = tosa.reciprocal %sum_broadcast : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %softmax = tosa.mul %exp, %reciprocal, %shift : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %softmax_f16 = tosa.cast %softmax : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf16>
  %softmax_collapsed = tensor.collapse_shape %softmax_f16 [[0, 1], [2], [3]] : tensor<1x2x1x8xf16> into tensor<2x1x8xf16>
  %values = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [2, 8, 2] : tensor<32xf16> into tensor<2x8x2xf16>
  %attention = tosa.matmul %softmax_collapsed, %values, %zero, %zero {acc_type = f32} : (tensor<2x1x8xf16>, tensor<2x8x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x2xf16>
  %attention_expanded = tensor.expand_shape %attention [[0, 1], [2], [3]] output_shape [1, 2, 1, 2] : tensor<2x1x2xf16> into tensor<1x2x1x2xf16>
  %attention_transposed = tosa.transpose %attention_expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x1x2xf16>) -> tensor<1x1x2x2xf16>
  %result = tensor.collapse_shape %attention_transposed [[0, 1, 2, 3]] : tensor<1x1x2x2xf16> into tensor<4xf16>
  return %result : tensor<4xf16>
}

// -----

// A sliding-window size larger than the key sequence length is rejected by the
// Rock verifier. Keep the lower mask explicit instead of creating invalid
// rock.attention IR.
// CHECK-LABEL: func @sliding_window_exceeds_max_seq_len
// CHECK: rock.attention
// CHECK: currentSeqLen =
// CHECK-NOT: slidingWindowSize
// CHECK: qk = elementwise
// CHECK: tosa.add
// CHECK: tosa.greater
// CHECK: tosa.select
func.func @sliding_window_exceeds_max_seq_len(%seq_len: tensor<1xi32>, %queries_flat: tensor<12xf16>, %keys_flat: tensor<32xf16>, %values_flat: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %softmax_ones = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x1x8xf32>}> : () -> tensor<1x2x1x8xf32>
  %mask_ones = "tosa.const"() <{values = dense<1> : tensor<1x2x1x8xi8>}> : () -> tensor<1x2x1x8xi8>
  %window_broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<8x1x1x1xi32>}> : () -> tensor<8x1x1x1xi32>
  %scale = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %neg_inf = "tosa.const"() <{values = dense<0xFC00> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %zero = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %window_offset = "tosa.const"() <{values = dense<-100> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %kv_broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<1x1x1x8xi32>}> : () -> tensor<1x1x1x8xi32>
  %columns_1d = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>}> : () -> tensor<8xi32>
  %columns_4d = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %keys_expanded = tensor.expand_shape %keys_flat [[0, 1, 2, 3]] output_shape [1, 2, 8, 2] : tensor<32xf16> into tensor<1x2x8x2xf16>
  %queries_expanded = tensor.expand_shape %queries_flat [[0, 1, 2, 3]] output_shape [1, 6, 1, 2] : tensor<12xf16> into tensor<1x6x1x2xf16>
  %seq_len_4d = tensor.expand_shape %seq_len [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %queries = tensor.extract_slice %queries_expanded[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %keys = tosa.transpose %keys_expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %queries_collapsed = tensor.collapse_shape %queries [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %keys_collapsed = tensor.collapse_shape %keys [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %scores = tosa.matmul %queries_collapsed, %keys_collapsed, %zero, %zero {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %scores_expanded = tensor.expand_shape %scores [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %scaled_scores = tosa.mul %scores_expanded, %scale, %shift : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %window_lower = tosa.add %seq_len_4d, %window_offset : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %window_broadcast = tosa.mul %window_lower, %window_broadcast_ones, %shift : (tensor<1x1x1x1xi32>, tensor<8x1x1x1xi32>, tensor<1xi8>) -> tensor<8x1x1x1xi32>
  %window_flat = tensor.collapse_shape %window_broadcast [[0, 1, 2, 3]] : tensor<8x1x1x1xi32> into tensor<8xi32>
  %window_pred = tosa.greater %window_flat, %columns_1d : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  %window_pred_i32 = tosa.cast %window_pred : (tensor<8xi1>) -> tensor<8xi32>
  %window_pred_i8 = tosa.cast %window_pred_i32 : (tensor<8xi32>) -> tensor<8xi8>
  %window_pred_4d = tensor.expand_shape %window_pred_i8 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %window_pred_broadcast = tosa.mul %window_pred_4d, %mask_ones, %shift : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %window_mask = tosa.cast %window_pred_broadcast : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %window_masked_scores = tosa.select %window_mask, %neg_inf, %scaled_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %kv_seq_len = tosa.mul %seq_len_4d, %kv_broadcast_ones, %shift : (tensor<1x1x1x1xi32>, tensor<1x1x1x8xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
  %kv_pred = tosa.greater %columns_4d, %kv_seq_len : (tensor<1x1x1x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi1>
  %kv_pred_i32 = tosa.cast %kv_pred : (tensor<1x1x1x8xi1>) -> tensor<1x1x1x8xi32>
  %kv_pred_i8 = tosa.cast %kv_pred_i32 : (tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi8>
  %kv_pred_broadcast = tosa.mul %kv_pred_i8, %mask_ones, %shift : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %kv_mask = tosa.cast %kv_pred_broadcast : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %masked_scores = tosa.select %kv_mask, %neg_inf, %window_masked_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %scores_f32 = tosa.cast %masked_scores : (tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf32>
  %max = tosa.reduce_max %scores_f32 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %max_broadcast = tosa.mul %max, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %normalized = tosa.sub %scores_f32, %max_broadcast : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %exp = tosa.exp %normalized : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %sum_broadcast = tosa.mul %sum, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %reciprocal = tosa.reciprocal %sum_broadcast : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %softmax = tosa.mul %exp, %reciprocal, %shift : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %softmax_f16 = tosa.cast %softmax : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf16>
  %softmax_collapsed = tensor.collapse_shape %softmax_f16 [[0, 1], [2], [3]] : tensor<1x2x1x8xf16> into tensor<2x1x8xf16>
  %values = tensor.expand_shape %values_flat [[0, 1, 2]] output_shape [2, 8, 2] : tensor<32xf16> into tensor<2x8x2xf16>
  %attention = tosa.matmul %softmax_collapsed, %values, %zero, %zero {acc_type = f32} : (tensor<2x1x8xf16>, tensor<2x8x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x2xf16>
  %attention_expanded = tensor.expand_shape %attention [[0, 1], [2], [3]] output_shape [1, 2, 1, 2] : tensor<2x1x2xf16> into tensor<1x2x1x2xf16>
  %attention_transposed = tosa.transpose %attention_expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x1x2xf16>) -> tensor<1x1x2x2xf16>
  %result = tensor.collapse_shape %attention_transposed [[0, 1, 2, 3]] : tensor<1x1x2x2xf16> into tensor<4xf16>
  return %result : tensor<4xf16>
}

// -----

// The column range may be split across an extra allowed dimension before
// collapsing to the key-sequence dimension. Validate the window against the
// first GEMM's key length (8), not the range tensor's trailing dimension (4).
// CHECK-LABEL: func @sliding_window_uses_key_seq_len
// CHECK: currentSeqLen =
// CHECK: slidingWindowSize = 6
// CHECK: qk = elementwise
// CHECK-NOT: tosa.greater
// CHECK-NOT: tosa.select
// CHECK: rock.yield
func.func @sliding_window_uses_key_seq_len(%seq_len: tensor<1xi32>, %queries_flat: tensor<12xf16>, %keys_flat: tensor<32xf16>, %values_flat: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %softmax_ones = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x1x8xf32>}> : () -> tensor<1x2x1x8xf32>
  %mask_ones = "tosa.const"() <{values = dense<1> : tensor<1x2x1x8xi8>}> : () -> tensor<1x2x1x8xi8>
  %window_broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<1x1x2x1x4xi32>}> : () -> tensor<1x1x2x1x4xi32>
  %scale = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %neg_inf = "tosa.const"() <{values = dense<0xFC00> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %zero = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %window_offset = "tosa.const"() <{values = dense<-6> : tensor<1x1x1x1x1xi32>}> : () -> tensor<1x1x1x1x1xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %kv_broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<1x1x1x8xi32>}> : () -> tensor<1x1x1x8xi32>
  %columns_split = "tosa.const"() <{values = dense<[[[[[0, 1, 2, 3]], [[4, 5, 6, 7]]]]]> : tensor<1x1x2x1x4xi32>}> : () -> tensor<1x1x2x1x4xi32>
  %columns_4d = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %keys_expanded = tensor.expand_shape %keys_flat [[0, 1, 2, 3]] output_shape [1, 2, 8, 2] : tensor<32xf16> into tensor<1x2x8x2xf16>
  %queries_expanded = tensor.expand_shape %queries_flat [[0, 1, 2, 3]] output_shape [1, 6, 1, 2] : tensor<12xf16> into tensor<1x6x1x2xf16>
  %seq_len_4d = tensor.expand_shape %seq_len [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %seq_len_5d = tensor.expand_shape %seq_len [[0, 1, 2, 3, 4]] output_shape [1, 1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1x1xi32>
  %queries = tensor.extract_slice %queries_expanded[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %keys = tosa.transpose %keys_expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %queries_collapsed = tensor.collapse_shape %queries [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %keys_collapsed = tensor.collapse_shape %keys [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %scores = tosa.matmul %queries_collapsed, %keys_collapsed, %zero, %zero {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %scores_expanded = tensor.expand_shape %scores [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %scaled_scores = tosa.mul %scores_expanded, %scale, %shift : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %window_lower = tosa.add %seq_len_5d, %window_offset : (tensor<1x1x1x1x1xi32>, tensor<1x1x1x1x1xi32>) -> tensor<1x1x1x1x1xi32>
  %window_broadcast = tosa.mul %window_lower, %window_broadcast_ones, %shift : (tensor<1x1x1x1x1xi32>, tensor<1x1x2x1x4xi32>, tensor<1xi8>) -> tensor<1x1x2x1x4xi32>
  %window_pred_split = tosa.greater %window_broadcast, %columns_split : (tensor<1x1x2x1x4xi32>, tensor<1x1x2x1x4xi32>) -> tensor<1x1x2x1x4xi1>
  %window_pred = tensor.collapse_shape %window_pred_split [[0, 1, 2, 3, 4]] : tensor<1x1x2x1x4xi1> into tensor<8xi1>
  %window_pred_i32 = tosa.cast %window_pred : (tensor<8xi1>) -> tensor<8xi32>
  %window_pred_i8 = tosa.cast %window_pred_i32 : (tensor<8xi32>) -> tensor<8xi8>
  %window_pred_4d = tensor.expand_shape %window_pred_i8 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %window_pred_broadcast = tosa.mul %window_pred_4d, %mask_ones, %shift : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %window_mask = tosa.cast %window_pred_broadcast : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %window_masked_scores = tosa.select %window_mask, %neg_inf, %scaled_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %kv_seq_len = tosa.mul %seq_len_4d, %kv_broadcast_ones, %shift : (tensor<1x1x1x1xi32>, tensor<1x1x1x8xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
  %kv_pred = tosa.greater %columns_4d, %kv_seq_len : (tensor<1x1x1x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi1>
  %kv_pred_i32 = tosa.cast %kv_pred : (tensor<1x1x1x8xi1>) -> tensor<1x1x1x8xi32>
  %kv_pred_i8 = tosa.cast %kv_pred_i32 : (tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi8>
  %kv_pred_broadcast = tosa.mul %kv_pred_i8, %mask_ones, %shift : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %kv_mask = tosa.cast %kv_pred_broadcast : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %masked_scores = tosa.select %kv_mask, %neg_inf, %window_masked_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %scores_f32 = tosa.cast %masked_scores : (tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf32>
  %max = tosa.reduce_max %scores_f32 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %max_broadcast = tosa.mul %max, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %normalized = tosa.sub %scores_f32, %max_broadcast : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %exp = tosa.exp %normalized : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %sum_broadcast = tosa.mul %sum, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %reciprocal = tosa.reciprocal %sum_broadcast : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %softmax = tosa.mul %exp, %reciprocal, %shift : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %softmax_f16 = tosa.cast %softmax : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf16>
  %softmax_collapsed = tensor.collapse_shape %softmax_f16 [[0, 1], [2], [3]] : tensor<1x2x1x8xf16> into tensor<2x1x8xf16>
  %values = tensor.expand_shape %values_flat [[0, 1, 2]] output_shape [2, 8, 2] : tensor<32xf16> into tensor<2x8x2xf16>
  %attention = tosa.matmul %softmax_collapsed, %values, %zero, %zero {acc_type = f32} : (tensor<2x1x8xf16>, tensor<2x8x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x2xf16>
  %attention_expanded = tensor.expand_shape %attention [[0, 1], [2], [3]] output_shape [1, 2, 1, 2] : tensor<2x1x2xf16> into tensor<1x2x1x2xf16>
  %attention_transposed = tosa.transpose %attention_expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x1x2xf16>) -> tensor<1x1x2x2xf16>
  %result = tensor.collapse_shape %attention_transposed [[0, 1, 2, 3]] : tensor<1x1x2x2xf16> into tensor<4xf16>
  return %result : tensor<4xf16>
}

// -----

// Negating INT32_MIN produces a window size that cannot be represented by the
// i32 slidingWindowSize attribute. Keep the lower mask explicit rather than
// narrowing it to a negative attribute. Since that unsupported mask is the
// outer select, keep the nested KV-cache mask explicit too; peeling it alone
// would require rebuilding the surrounding select chain.
// CHECK-LABEL: func @sliding_window_int32_min_offset
// CHECK: rock.attention
// CHECK-NOT: currentSeqLen
// CHECK-NOT: slidingWindowSize
// CHECK: qk = elementwise
// CHECK: tosa.add
// CHECK: tosa.greater
// CHECK: tosa.select
func.func @sliding_window_int32_min_offset(%seq_len: tensor<1xi32>, %queries_flat: tensor<12xf16>, %keys_flat: tensor<32xf16>, %values_flat: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %softmax_ones = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x1x8xf32>}> : () -> tensor<1x2x1x8xf32>
  %mask_ones = "tosa.const"() <{values = dense<1> : tensor<1x2x1x8xi8>}> : () -> tensor<1x2x1x8xi8>
  %window_broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<8x1x1x1xi32>}> : () -> tensor<8x1x1x1xi32>
  %scale = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %neg_inf = "tosa.const"() <{values = dense<0xFC00> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %zero = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %window_offset = "tosa.const"() <{values = dense<-2147483648> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %kv_broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<1x1x1x8xi32>}> : () -> tensor<1x1x1x8xi32>
  %columns_1d = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>}> : () -> tensor<8xi32>
  %columns_4d = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %keys_expanded = tensor.expand_shape %keys_flat [[0, 1, 2, 3]] output_shape [1, 2, 8, 2] : tensor<32xf16> into tensor<1x2x8x2xf16>
  %queries_expanded = tensor.expand_shape %queries_flat [[0, 1, 2, 3]] output_shape [1, 6, 1, 2] : tensor<12xf16> into tensor<1x6x1x2xf16>
  %seq_len_4d = tensor.expand_shape %seq_len [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi32> into tensor<1x1x1x1xi32>
  %queries = tensor.extract_slice %queries_expanded[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %keys = tosa.transpose %keys_expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %queries_collapsed = tensor.collapse_shape %queries [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %keys_collapsed = tensor.collapse_shape %keys [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %scores = tosa.matmul %queries_collapsed, %keys_collapsed, %zero, %zero {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %scores_expanded = tensor.expand_shape %scores [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %scaled_scores = tosa.mul %scores_expanded, %scale, %shift : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %window_lower = tosa.add %seq_len_4d, %window_offset : (tensor<1x1x1x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  %window_broadcast = tosa.mul %window_lower, %window_broadcast_ones, %shift : (tensor<1x1x1x1xi32>, tensor<8x1x1x1xi32>, tensor<1xi8>) -> tensor<8x1x1x1xi32>
  %window_flat = tensor.collapse_shape %window_broadcast [[0, 1, 2, 3]] : tensor<8x1x1x1xi32> into tensor<8xi32>
  %window_pred = tosa.greater %window_flat, %columns_1d : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  %window_pred_i32 = tosa.cast %window_pred : (tensor<8xi1>) -> tensor<8xi32>
  %window_pred_i8 = tosa.cast %window_pred_i32 : (tensor<8xi32>) -> tensor<8xi8>
  %window_pred_4d = tensor.expand_shape %window_pred_i8 [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xi8> into tensor<1x1x1x8xi8>
  %window_pred_broadcast = tosa.mul %window_pred_4d, %mask_ones, %shift : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %window_mask = tosa.cast %window_pred_broadcast : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %kv_seq_len = tosa.mul %seq_len_4d, %kv_broadcast_ones, %shift : (tensor<1x1x1x1xi32>, tensor<1x1x1x8xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
  %kv_pred = tosa.greater %columns_4d, %kv_seq_len : (tensor<1x1x1x8xi32>, tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi1>
  %kv_pred_i32 = tosa.cast %kv_pred : (tensor<1x1x1x8xi1>) -> tensor<1x1x1x8xi32>
  %kv_pred_i8 = tosa.cast %kv_pred_i32 : (tensor<1x1x1x8xi32>) -> tensor<1x1x1x8xi8>
  %kv_pred_broadcast = tosa.mul %kv_pred_i8, %mask_ones, %shift : (tensor<1x1x1x8xi8>, tensor<1x2x1x8xi8>, tensor<1xi8>) -> tensor<1x2x1x8xi8>
  %kv_mask = tosa.cast %kv_pred_broadcast : (tensor<1x2x1x8xi8>) -> tensor<1x2x1x8xi1>
  %kv_masked_scores = tosa.select %kv_mask, %neg_inf, %scaled_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %masked_scores = tosa.select %window_mask, %neg_inf, %kv_masked_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %scores_f32 = tosa.cast %masked_scores : (tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf32>
  %max = tosa.reduce_max %scores_f32 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %max_broadcast = tosa.mul %max, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %normalized = tosa.sub %scores_f32, %max_broadcast : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %exp = tosa.exp %normalized : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %sum_broadcast = tosa.mul %sum, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %reciprocal = tosa.reciprocal %sum_broadcast : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %softmax = tosa.mul %exp, %reciprocal, %shift : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %softmax_f16 = tosa.cast %softmax : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf16>
  %softmax_collapsed = tensor.collapse_shape %softmax_f16 [[0, 1], [2], [3]] : tensor<1x2x1x8xf16> into tensor<2x1x8xf16>
  %values = tensor.expand_shape %values_flat [[0, 1, 2]] output_shape [2, 8, 2] : tensor<32xf16> into tensor<2x8x2xf16>
  %attention = tosa.matmul %softmax_collapsed, %values, %zero, %zero {acc_type = f32} : (tensor<2x1x8xf16>, tensor<2x8x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x2xf16>
  %attention_expanded = tensor.expand_shape %attention [[0, 1], [2], [3]] output_shape [1, 2, 1, 2] : tensor<2x1x2xf16> into tensor<1x2x1x2xf16>
  %attention_transposed = tosa.transpose %attention_expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x2x1x2xf16>) -> tensor<1x1x2x2xf16>
  %result = tensor.collapse_shape %attention_transposed [[0, 1, 2, 3]] : tensor<1x1x2x2xf16> into tensor<4xf16>
  return %result : tensor<4xf16>
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
