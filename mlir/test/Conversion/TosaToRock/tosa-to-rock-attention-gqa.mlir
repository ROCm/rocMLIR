// RUN: rocmlir-opt --tosa-to-rock %s -verify-diagnostics -o -| FileCheck %s

// CHECK: rock.attention
func.func @self_attention_gqa(%arg0: tensor<4096xf32> {mhal.read_access}, %arg1: tensor<8192xf32> {mhal.read_access}, %arg2: tensor<4096xf32> {mhal.read_access}) -> (tensor<8192xf32> {mhal.write_access}) attributes {kernel, arch = ""} {
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 4, 32, 32] : tensor<8192xf32> into tensor<2x4x32x32xf32>
  %expanded_0 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x2x2x32x32xf32>}> : () -> tensor<2x2x2x32x32xf32>
  %1 = tosa.add %expanded_2, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %expanded_3 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %2 = tosa.add %expanded_3, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed_4 = tensor.collapse_shape %2 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %4 = tosa.transpose %collapsed_4 {perms = array<i32: 0, 1, 3, 2>} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %expanded_5 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [8, 32, 32] : tensor<8192xf32> into tensor<8x32x32xf32>
  %collapsed_6 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %5 = tosa.matmul %expanded_5, %collapsed_6 : (tensor<8x32x32xf32>, tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
  %expanded_7 = tensor.expand_shape %5 [[0, 1], [2], [3]] output_shape [2, 4, 32, 32] : tensor<8x32x32xf32> into tensor<2x4x32x32xf32>
  %6 = tosa.reduce_max %expanded_7 {axis = 3 : i32} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x1xf32>
  %7 = tosa.sub %expanded_7, %6 : (tensor<2x4x32x32xf32>, tensor<2x4x32x1xf32>) -> tensor<2x4x32x32xf32>
  %8 = tosa.exp %7 : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %9 = tosa.reduce_sum %8 {axis = 3 : i32} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x1xf32>
  %10 = tosa.reciprocal %9 : (tensor<2x4x32x1xf32>) -> tensor<2x4x32x1xf32>
  %shift = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %11 = tosa.mul %8, %10, %shift : (tensor<2x4x32x32xf32>, tensor<2x4x32x1xf32>, tensor<1xi8>) -> tensor<2x4x32x32xf32>
  %collapsed_8 = tensor.collapse_shape %11 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %collapsed_9 = tensor.collapse_shape %1 [[0, 1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<8x32x32xf32>
  %12 = tosa.matmul %collapsed_8, %collapsed_9 : (tensor<8x32x32xf32>, tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
  %expanded_10 = tensor.expand_shape %12 [[0, 1], [2], [3]] output_shape [2, 4, 32, 32] : tensor<8x32x32xf32> into tensor<2x4x32x32xf32>
  %collapsed_11 = tensor.collapse_shape %12 [[0, 1, 2]] : tensor<8x32x32xf32> into tensor<8192xf32>
  return %collapsed_11 : tensor<8192xf32>
}


// CHECK: rock.attention
func.func @self_attention_gqa_bias(%arg0: tensor<4096xf32> {mhal.read_access}, %arg1: tensor<8192xf32> {mhal.read_access}, %arg2: tensor<4096xf32> {mhal.read_access}, %arg3: tensor<8192xf32> {mhal.read_access}) -> (tensor<8192xf32> {mhal.write_access}) attributes {kernel, arch = ""} {
  %expanded = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [2, 4, 32, 32] : tensor<8192xf32> into tensor<2x4x32x32xf32>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 4, 32, 32] : tensor<8192xf32> into tensor<2x4x32x32xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_3 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x2x2x32x32xf32>}> : () -> tensor<2x2x2x32x32xf32>
  %1 = tosa.add %expanded_3, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %expanded_4 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %2 = tosa.add %expanded_4, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed_5 = tensor.collapse_shape %2 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %3 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %4 = tosa.transpose %collapsed_5 {perms = array<i32: 0, 1, 3, 2>} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %expanded_6 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [8, 32, 32] : tensor<8192xf32> into tensor<8x32x32xf32>
  %collapsed_7 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %5 = tosa.matmul %expanded_6, %collapsed_7 : (tensor<8x32x32xf32>, tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
  %expanded_8 = tensor.expand_shape %5 [[0, 1], [2], [3]] output_shape [2, 4, 32, 32] : tensor<8x32x32xf32> into tensor<2x4x32x32xf32>
  %6 = tosa.add %expanded_8, %expanded : (tensor<2x4x32x32xf32>, tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %7 = tosa.reduce_max %6 {axis = 3 : i32} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x1xf32>
  %8 = tosa.sub %6, %7 : (tensor<2x4x32x32xf32>, tensor<2x4x32x1xf32>) -> tensor<2x4x32x32xf32>
  %9 = tosa.exp %8 : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %10 = tosa.reduce_sum %9 {axis = 3 : i32} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x1xf32>
  %11 = tosa.reciprocal %10 : (tensor<2x4x32x1xf32>) -> tensor<2x4x32x1xf32>
  %shift = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %12 = tosa.mul %9, %11, %shift : (tensor<2x4x32x32xf32>, tensor<2x4x32x1xf32>, tensor<1xi8>) -> tensor<2x4x32x32xf32>
  %collapsed_9 = tensor.collapse_shape %12 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %collapsed_10 = tensor.collapse_shape %1 [[0, 1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<8x32x32xf32>
  %13 = tosa.matmul %collapsed_9, %collapsed_10 : (tensor<8x32x32xf32>, tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
  %expanded_11 = tensor.expand_shape %13 [[0, 1], [2], [3]] output_shape [2, 4, 32, 32] : tensor<8x32x32xf32> into tensor<2x4x32x32xf32>
  %collapsed_12 = tensor.collapse_shape %13 [[0, 1, 2]] : tensor<8x32x32xf32> into tensor<8192xf32>
  return %collapsed_12 : tensor<8192xf32>
}

// CHECK: rock.attention
func.func @self_attention_gqa_scale(%arg0: tensor<4096xf32> {mhal.read_access}, %arg1: tensor<8192xf32> {mhal.read_access}, %arg2: tensor<8192xf32> {mhal.read_access}, %arg3: tensor<4096xf32> {mhal.read_access}) -> (tensor<8192xf32> {mhal.write_access}) attributes {kernel, arch = ""} {
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 4, 32, 32] : tensor<8192xf32> into tensor<2x4x32x32xf32>
  %expanded_0 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [2, 4, 32, 32] : tensor<8192xf32> into tensor<2x4x32x32xf32>
  %expanded_1 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_3 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x2x2x32x32xf32>}> : () -> tensor<2x2x2x32x32xf32>
  %1 = tosa.add %expanded_3, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %expanded_4 = tensor.expand_shape %arg3 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %2 = tosa.add %expanded_4, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed_5 = tensor.collapse_shape %2 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %4 = tosa.transpose %collapsed_5 {perms = array<i32: 0, 1, 3, 2>} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %expanded_6 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [8, 32, 32] : tensor<8192xf32> into tensor<8x32x32xf32>
  %collapsed_7 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %5 = tosa.matmul %expanded_6, %collapsed_7 : (tensor<8x32x32xf32>, tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
  %expanded_8 = tensor.expand_shape %5 [[0, 1], [2], [3]] output_shape [2, 4, 32, 32] : tensor<8x32x32xf32> into tensor<2x4x32x32xf32>
  %shift = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %6 = tosa.mul %expanded_8, %expanded, %shift : (tensor<2x4x32x32xf32>, tensor<2x4x32x32xf32>, tensor<1xi8>) -> tensor<2x4x32x32xf32>
  %7 = tosa.reduce_max %6 {axis = 3 : i32} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x1xf32>
  %8 = tosa.sub %6, %7 : (tensor<2x4x32x32xf32>, tensor<2x4x32x1xf32>) -> tensor<2x4x32x32xf32>
  %9 = tosa.exp %8 : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %10 = tosa.reduce_sum %9 {axis = 3 : i32} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x1xf32>
  %11 = tosa.reciprocal %10 : (tensor<2x4x32x1xf32>) -> tensor<2x4x32x1xf32>
  %12 = tosa.mul %9, %11, %shift : (tensor<2x4x32x32xf32>, tensor<2x4x32x1xf32>, tensor<1xi8>) -> tensor<2x4x32x32xf32>
  %collapsed_9 = tensor.collapse_shape %12 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %collapsed_10 = tensor.collapse_shape %1 [[0, 1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<8x32x32xf32>
  %13 = tosa.matmul %collapsed_9, %collapsed_10 : (tensor<8x32x32xf32>, tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
  %expanded_11 = tensor.expand_shape %13 [[0, 1], [2], [3]] output_shape [2, 4, 32, 32] : tensor<8x32x32xf32> into tensor<2x4x32x32xf32>
  %collapsed_12 = tensor.collapse_shape %13 [[0, 1, 2]] : tensor<8x32x32xf32> into tensor<8192xf32>
  return %collapsed_12 : tensor<8192xf32>
}

// CHECK: rock.attention
func.func @self_attention_gqa_scale_bias(%arg0: tensor<4096xf32> {mhal.read_access}, %arg1: tensor<8192xf32> {mhal.read_access}, %arg2: tensor<8192xf32> {mhal.read_access}, %arg3: tensor<4096xf32> {mhal.read_access}, %arg4: tensor<8192xf32> {mhal.read_access}) -> (tensor<8192xf32> {mhal.write_access}) attributes {kernel, arch = ""} {
  %expanded = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [2, 4, 32, 32] : tensor<8192xf32> into tensor<2x4x32x32xf32>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 4, 32, 32] : tensor<8192xf32> into tensor<2x4x32x32xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [2, 4, 32, 32] : tensor<8192xf32> into tensor<2x4x32x32xf32>
  %expanded_2 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_3 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_4 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x2x2x32x32xf32>}> : () -> tensor<2x2x2x32x32xf32>
  %1 = tosa.add %expanded_4, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %expanded_5 = tensor.expand_shape %arg3 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %2 = tosa.add %expanded_5, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed_6 = tensor.collapse_shape %2 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %4 = tosa.transpose %collapsed_6 {perms = array<i32: 0, 1, 3, 2>} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %expanded_7 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [8, 32, 32] : tensor<8192xf32> into tensor<8x32x32xf32>
  %collapsed_8 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %5 = tosa.matmul %expanded_7, %collapsed_8 : (tensor<8x32x32xf32>, tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
  %expanded_9 = tensor.expand_shape %5 [[0, 1], [2], [3]] output_shape [2, 4, 32, 32] : tensor<8x32x32xf32> into tensor<2x4x32x32xf32>
  %shift = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %6 = tosa.mul %expanded_9, %expanded_0, %shift : (tensor<2x4x32x32xf32>, tensor<2x4x32x32xf32>, tensor<1xi8>) -> tensor<2x4x32x32xf32>
  %7 = tosa.add %6, %expanded : (tensor<2x4x32x32xf32>, tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %8 = tosa.reduce_max %7 {axis = 3 : i32} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x1xf32>
  %9 = tosa.sub %7, %8 : (tensor<2x4x32x32xf32>, tensor<2x4x32x1xf32>) -> tensor<2x4x32x32xf32>
  %10 = tosa.exp %9 : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %11 = tosa.reduce_sum %10 {axis = 3 : i32} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x1xf32>
  %12 = tosa.reciprocal %11 : (tensor<2x4x32x1xf32>) -> tensor<2x4x32x1xf32>
  %13 = tosa.mul %10, %12, %shift : (tensor<2x4x32x32xf32>, tensor<2x4x32x1xf32>, tensor<1xi8>) -> tensor<2x4x32x32xf32>
  %collapsed_10 = tensor.collapse_shape %13 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %collapsed_11 = tensor.collapse_shape %1 [[0, 1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<8x32x32xf32>
  %14 = tosa.matmul %collapsed_10, %collapsed_11 : (tensor<8x32x32xf32>, tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
  %expanded_12 = tensor.expand_shape %14 [[0, 1], [2], [3]] output_shape [2, 4, 32, 32] : tensor<8x32x32xf32> into tensor<2x4x32x32xf32>
  %collapsed_13 = tensor.collapse_shape %14 [[0, 1, 2]] : tensor<8x32x32xf32> into tensor<8192xf32>
  return %collapsed_13 : tensor<8192xf32>
}

// CHECK: rock.attention
func.func @self_attention_gqa_scale_bias_kvcache(%arg0: tensor<4096xf32> {mhal.read_access}, %arg1: tensor<256xf32> {mhal.read_access}, %arg2: tensor<256xf32> {mhal.read_access}, %arg3: tensor<4096xf32> {mhal.read_access}, %arg4: tensor<256xf32> {mhal.read_access}) -> (tensor<256xf32> {mhal.write_access}) attributes {kernel, arch = ""} {
  %expanded = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [2, 4, 1, 32] : tensor<256xf32> into tensor<2x4x1x32xf32>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 4, 1, 32] : tensor<256xf32> into tensor<2x4x1x32xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [2, 4, 1, 32] : tensor<256xf32> into tensor<2x4x1x32xf32>
  %expanded_2 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_3 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 2, 32, 32] : tensor<4096xf32> into tensor<2x2x32x32xf32>
  %expanded_4 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %0 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x2x2x32x32xf32>}> : () -> tensor<2x2x2x32x32xf32>
  %1 = tosa.add %expanded_4, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %expanded_5 = tensor.expand_shape %arg3 [[0, 1, 2, 3, 4]] output_shape [2, 2, 1, 32, 32] : tensor<4096xf32> into tensor<2x2x1x32x32xf32>
  %2 = tosa.add %expanded_5, %0 : (tensor<2x2x1x32x32xf32>, tensor<2x2x2x32x32xf32>) -> tensor<2x2x2x32x32xf32>
  %collapsed_6 = tensor.collapse_shape %2 [[0], [1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<2x4x32x32xf32>
  %4 = tosa.transpose %collapsed_6 {perms = array<i32: 0, 1, 3, 2>} : (tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32>
  %expanded_7 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [8, 1, 32] : tensor<256xf32> into tensor<8x1x32xf32>
  %collapsed_8 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<2x4x32x32xf32> into tensor<8x32x32xf32>
  %5 = tosa.matmul %expanded_7, %collapsed_8 : (tensor<8x1x32xf32>, tensor<8x32x32xf32>) -> tensor<8x1x32xf32>
  %expanded_9 = tensor.expand_shape %5 [[0, 1], [2], [3]] output_shape [2, 4, 1, 32] : tensor<8x1x32xf32> into tensor<2x4x1x32xf32>
  %shift = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %6 = tosa.mul %expanded_9, %expanded_0, %shift : (tensor<2x4x1x32xf32>, tensor<2x4x1x32xf32>, tensor<1xi8>) -> tensor<2x4x1x32xf32>
  %7 = tosa.add %6, %expanded : (tensor<2x4x1x32xf32>, tensor<2x4x1x32xf32>) -> tensor<2x4x1x32xf32>
  %8 = tosa.reduce_max %7 {axis = 3 : i32} : (tensor<2x4x1x32xf32>) -> tensor<2x4x1x1xf32>
  %9 = tosa.sub %7, %8 : (tensor<2x4x1x32xf32>, tensor<2x4x1x1xf32>) -> tensor<2x4x1x32xf32>
  %10 = tosa.exp %9 : (tensor<2x4x1x32xf32>) -> tensor<2x4x1x32xf32>
  %11 = tosa.reduce_sum %10 {axis = 3 : i32} : (tensor<2x4x1x32xf32>) -> tensor<2x4x1x1xf32>
  %12 = tosa.reciprocal %11 : (tensor<2x4x1x1xf32>) -> tensor<2x4x1x1xf32>
  %13 = tosa.mul %10, %12, %shift : (tensor<2x4x1x32xf32>, tensor<2x4x1x1xf32>, tensor<1xi8>) -> tensor<2x4x1x32xf32>
  %collapsed_10 = tensor.collapse_shape %13 [[0, 1], [2], [3]] : tensor<2x4x1x32xf32> into tensor<8x1x32xf32>
  %collapsed_11 = tensor.collapse_shape %1 [[0, 1, 2], [3], [4]] : tensor<2x2x2x32x32xf32> into tensor<8x32x32xf32>
  %14 = tosa.matmul %collapsed_10, %collapsed_11 : (tensor<8x1x32xf32>, tensor<8x32x32xf32>) -> tensor<8x1x32xf32>
  %expanded_12 = tensor.expand_shape %14 [[0, 1], [2], [3]] output_shape [2, 4, 1, 32] : tensor<8x1x32xf32> into tensor<2x4x1x32xf32>
  %collapsed_13 = tensor.collapse_shape %14 [[0, 1, 2]] : tensor<8x1x32xf32> into tensor<256xf32>
  return %collapsed_13 : tensor<256xf32>
}
