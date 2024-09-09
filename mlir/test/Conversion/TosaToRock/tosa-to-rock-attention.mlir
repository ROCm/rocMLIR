// RUN: rocmlir-opt --tosa-to-rock %s -verify-diagnostics -o -| FileCheck %s

// CHECK: rock.attention
func.func @self_attention(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = ""} {
  %cst = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
  %0 = "tosa.transpose"(%arg1, %cst) : (tensor<1x384x64xf32>, tensor<3xi64>) -> tensor<1x64x384xf32>
  %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>) -> tensor<1x384x384xf32>
  %2 = "tosa.mul"(%1, %arg3) {shift = 0 : i8} : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %3 = "tosa.reduce_max"(%2) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %4 = "tosa.sub"(%2, %3) : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %5 = "tosa.exp"(%4) : (tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %6 = "tosa.reduce_sum"(%5) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %7 = "tosa.reciprocal"(%6) : (tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
  %8 = "tosa.mul"(%5, %7) {shift = 0 : i8} : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %9 = "tosa.matmul"(%8, %arg2) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
  return %9 : tensor<1x384x64xf32>
}

// CHECK: rock.attention
func.func @self_attention_no_scale(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = ""} {
  %cst = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
  %0 = "tosa.transpose"(%arg1, %cst) : (tensor<1x384x64xf32>, tensor<3xi64>) -> tensor<1x64x384xf32>
  %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>) -> tensor<1x384x384xf32>
  %3 = "tosa.reduce_max"(%1) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %4 = "tosa.sub"(%1, %3) : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %5 = "tosa.exp"(%4) : (tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %6 = "tosa.reduce_sum"(%5) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %7 = "tosa.reciprocal"(%6) : (tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
  %8 = "tosa.mul"(%5, %7) {shift = 0 : i8} : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %9 = "tosa.matmul"(%8, %arg2) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
  return %9 : tensor<1x384x64xf32>
}

// CHECK: rock.attention
func.func @self_attention_with_bias_only(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = ""} {
  %cst = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
  %0 = "tosa.transpose"(%arg1, %cst) : (tensor<1x384x64xf32>, tensor<3xi64>) -> tensor<1x64x384xf32>
  %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>) -> tensor<1x384x384xf32>
  %2 = "tosa.add"(%1, %arg3) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %3 = "tosa.reduce_max"(%2) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %4 = "tosa.sub"(%2, %3) : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %5 = "tosa.exp"(%4) : (tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %6 = "tosa.reduce_sum"(%5) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %7 = "tosa.reciprocal"(%6) : (tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
  %8 = "tosa.mul"(%5, %7) {shift = 0 : i8} : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %9 = "tosa.matmul"(%8, %arg2) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
  return %9 : tensor<1x384x64xf32>
}

// CHECK: rock.attention
func.func @self_attention_with_scale_and_bias(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>, %arg4: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = ""} {
  %cst = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
  %0 = "tosa.transpose"(%arg1, %cst) : (tensor<1x384x64xf32>, tensor<3xi64>) -> tensor<1x64x384xf32>
  %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>) -> tensor<1x384x384xf32>
  %2 = "tosa.mul"(%1, %arg3) {shift = 0 : i8} : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %3 = "tosa.add"(%2, %arg4) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %4 = "tosa.reduce_max"(%3) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %5 = "tosa.sub"(%3, %4) : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %6 = "tosa.exp"(%5) : (tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %7 = "tosa.reduce_sum"(%6) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %8 = "tosa.reciprocal"(%7) : (tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
  %9 = "tosa.mul"(%6, %8) {shift = 0 : i8} : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %10 = "tosa.matmul"(%9, %arg2) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
  return %10 : tensor<1x384x64xf32>
}

// CHECK: rock.attention
func.func @self_attention_with_scale_bias_exp(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>, %arg4: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = ""} {
  %cst = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
  %0 = "tosa.transpose"(%arg1, %cst) : (tensor<1x384x64xf32>, tensor<3xi64>) -> tensor<1x64x384xf32>
  %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>) -> tensor<1x384x384xf32>
  %2 = "tosa.mul"(%1, %arg3) {shift = 0 : i8} : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %3 = "tosa.add"(%2, %arg4) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %exp = "tosa.exp"(%3) : (tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %4 = "tosa.reduce_max"(%exp) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %5 = "tosa.sub"(%exp, %4) : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %6 = "tosa.exp"(%5) : (tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %7 = "tosa.reduce_sum"(%6) {axis = 1 : i32} : (tensor<1x384x384xf32>) -> tensor<1x1x384xf32>
  %8 = "tosa.reciprocal"(%7) : (tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
  %9 = "tosa.mul"(%6, %8) {shift = 0 : i8} : (tensor<1x384x384xf32>, tensor<1x1x384xf32>) -> tensor<1x384x384xf32>
  %10 = "tosa.matmul"(%9, %arg2) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
  return %10 : tensor<1x384x64xf32>
}

// CHECK: rock.attention
func.func @self_attention_with_reshapes(%arg0: tensor<1x12x384x64xf32>, %arg1: tensor<1x12x64x384xf32>, %arg2: tensor<1x12x384x64xf32>) -> (tensor<1x12x384x64xf32>) attributes {kernel, arch = ""} {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x12x384x64xf32> into tensor<12x384x64xf32>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2], [3]] : tensor<1x12x64x384xf32> into tensor<12x64x384xf32>
  %0 = "tosa.matmul"(%collapsed, %collapsed_0) : (tensor<12x384x64xf32>, tensor<12x64x384xf32>) -> tensor<12x384x384xf32>
  %expanded = tensor.expand_shape %0 [[0, 1], [2], [3]] output_shape [1, 12, 384, 384] : tensor<12x384x384xf32> into tensor<1x12x384x384xf32>
  %1 = "tosa.reduce_max"(%expanded) <{axis = 3 : i32}> : (tensor<1x12x384x384xf32>) -> tensor<1x12x384x1xf32>
  %2 = "tosa.sub"(%expanded, %1) : (tensor<1x12x384x384xf32>, tensor<1x12x384x1xf32>) -> tensor<1x12x384x384xf32>
  %3 = "tosa.exp"(%2) : (tensor<1x12x384x384xf32>) -> tensor<1x12x384x384xf32>
  %4 = "tosa.reduce_sum"(%3) <{axis = 3 : i32}> : (tensor<1x12x384x384xf32>) -> tensor<1x12x384x1xf32>
  %5 = "tosa.reciprocal"(%4) : (tensor<1x12x384x1xf32>) -> tensor<1x12x384x1xf32>
  %6 = "tosa.mul"(%3, %5) <{shift = 0 : i8}> : (tensor<1x12x384x384xf32>, tensor<1x12x384x1xf32>) -> tensor<1x12x384x384xf32>
  %collapsed_1 = tensor.collapse_shape %6 [[0, 1], [2], [3]] : tensor<1x12x384x384xf32> into tensor<12x384x384xf32>
  %collapsed_2 = tensor.collapse_shape %arg2 [[0, 1], [2], [3]] : tensor<1x12x384x64xf32> into tensor<12x384x64xf32>
  %7 = "tosa.matmul"(%collapsed_1, %collapsed_2) : (tensor<12x384x384xf32>, tensor<12x384x64xf32>) -> tensor<12x384x64xf32>
  %expanded_3 = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 12, 384, 64] : tensor<12x384x64xf32> into tensor<1x12x384x64xf32>
  return %expanded_3 : tensor<1x12x384x64xf32>
}

// CHECK: rock.attention
func.func @self_attention_with_4d_scale(%arg0: tensor<1x12x256x256xf32> , %arg1: tensor<1x12x256x256xf32>, %arg2: tensor<1x12x256x256xf32>, %arg3: tensor<1x12x256x256xf32>) -> (tensor<1x12x256x256xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  %cst = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi64>
  %0 = "tosa.transpose"(%arg3, %cst) : (tensor<1x12x256x256xf32>, tensor<4xi64>) -> tensor<1x12x256x256xf32>
  %collapsed = tensor.collapse_shape %arg2 [[0, 1], [2], [3]] : tensor<1x12x256x256xf32> into tensor<12x256x256xf32>
  %collapsed_0 = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<1x12x256x256xf32> into tensor<12x256x256xf32>
  %1 = "tosa.matmul"(%collapsed, %collapsed_0) : (tensor<12x256x256xf32>, tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf32> into tensor<1x12x256x256xf32>
  %2 = "tosa.mul"(%expanded, %arg1) <{shift = 0 : i8}> : (tensor<1x12x256x256xf32>, tensor<1x12x256x256xf32>) -> tensor<1x12x256x256xf32>
  %3 = "tosa.reduce_max"(%2) <{axis = 3 : i32}> : (tensor<1x12x256x256xf32>) -> tensor<1x12x256x1xf32>
  %4 = "tosa.sub"(%2, %3) : (tensor<1x12x256x256xf32>, tensor<1x12x256x1xf32>) -> tensor<1x12x256x256xf32>
  %5 = "tosa.exp"(%4) : (tensor<1x12x256x256xf32>) -> tensor<1x12x256x256xf32>
  %6 = "tosa.reduce_sum"(%5) <{axis = 3 : i32}> : (tensor<1x12x256x256xf32>) -> tensor<1x12x256x1xf32>
  %7 = "tosa.reciprocal"(%6) : (tensor<1x12x256x1xf32>) -> tensor<1x12x256x1xf32>
  %8 = "tosa.mul"(%5, %7) <{shift = 0 : i8}> : (tensor<1x12x256x256xf32>, tensor<1x12x256x1xf32>) -> tensor<1x12x256x256xf32>
  %collapsed_1 = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x12x256x256xf32> into tensor<12x256x256xf32>
  %collapsed_2 = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x12x256x256xf32> into tensor<12x256x256xf32>
  %9 = "tosa.matmul"(%collapsed_1, %collapsed_2) : (tensor<12x256x256xf32>, tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %expanded_3 = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf32> into tensor<1x12x256x256xf32>
  return %expanded_3 : tensor<1x12x256x256xf32>
}

// CHECK: rock.attention
func.func @self_attention_with_dot_product(%arg0: tensor<1x1x64xf32>, %arg1: tensor<1x1x64xf32>, %arg2: tensor<1x1x64xf32>, %arg3: tensor<1x1x1xf32>, %arg4: tensor<1x1x1xf32>) -> tensor<1x1x64xf32> attributes {kernel, arch = ""} {
  %cst = arith.constant dense<[0, 2, 1]> : tensor<3xi64>
  %0 = "tosa.transpose"(%arg1, %cst) : (tensor<1x1x64xf32>, tensor<3xi64>) -> tensor<1x64x1xf32>
  %1 = "tosa.matmul"(%arg0, %0) : (tensor<1x1x64xf32>, tensor<1x64x1xf32>) -> tensor<1x1x1xf32>
  %2 = "tosa.mul"(%1, %arg3) {shift = 0 : i8} : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %3 = "tosa.add"(%2, %arg4) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %4 = "tosa.sub"(%3, %3) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %5 = "tosa.exp"(%4) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %6 = "tosa.reciprocal"(%5) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %7 = "tosa.mul"(%5, %6) {shift = 0 : i8} : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %8 = "tosa.matmul"(%7, %arg2) : (tensor<1x1x1xf32>, tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
  return %8 : tensor<1x1x64xf32>
}

// CHECK: rock.attention
// CHECK: elementwise otherIns(%arg2, %arg3 : tensor<786432xi8>, tensor<786432xf16>)
// CHECK: firstGemmIdx = 1 : i32
func.func @mlir_attention_where(%arg0: tensor<786432xf16>, %arg1: tensor<786432xf16>, %arg2: tensor<786432xi8>, %arg3: tensor<786432xf16>, %arg4: tensor<786432xf16>) -> tensor<786432xf16> attributes {arch = "gfx942", kernel = "mixr"} {
  %expanded = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xi8> into tensor<1x12x256x256xi8>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_3 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %0 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi64>}> : () -> tensor<4xi64>
  %1 = tosa.transpose %expanded_3, %0 : (tensor<1x12x256x256xf16>, tensor<4xi64>) -> tensor<1x12x256x256xf16>
  %expanded_4 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %2 = tosa.matmul %expanded_4, %collapsed : (tensor<12x256x256xf16>, tensor<12x256x256xf16>) -> tensor<12x256x256xf16>
  %expanded_5 = tensor.expand_shape %2 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf16> into tensor<1x12x256x256xf16>
  %3 = "tosa.const"() <{value = dense<1.250000e-01> : tensor<1x12x256x256xf16>}> : () -> tensor<1x12x256x256xf16>
  %4 = tosa.mul %expanded_5, %3 {shift = 0 : i8} : (tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %5 = tosa.cast %expanded_1 : (tensor<1x12x256x256xi8>) -> tensor<1x12x256x256xi1>
  %6 = tosa.select %5, %4, %expanded_0 : (tensor<1x12x256x256xi1>, tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %7 = tosa.reduce_max %6 {axis = 3 : i32} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x1xf16>
  %8 = tosa.sub %6, %7 : (tensor<1x12x256x256xf16>, tensor<1x12x256x1xf16>) -> tensor<1x12x256x256xf16>
  %9 = tosa.exp %8 : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %10 = tosa.reduce_sum %9 {axis = 3 : i32} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x1xf16>
  %11 = tosa.reciprocal %10 : (tensor<1x12x256x1xf16>) -> tensor<1x12x256x1xf16>
  %12 = tosa.mul %9, %11 {shift = 0 : i8} : (tensor<1x12x256x256xf16>, tensor<1x12x256x1xf16>) -> tensor<1x12x256x256xf16>
  %collapsed_6 = tensor.collapse_shape %12 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %expanded_7 = tensor.expand_shape %arg4 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %13 = tosa.matmul %collapsed_6, %expanded_7 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>) -> tensor<12x256x256xf16>
  %expanded_8 = tensor.expand_shape %13 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf16> into tensor<1x12x256x256xf16>
  %collapsed_9 = tensor.collapse_shape %13 [[0, 1, 2]] : tensor<12x256x256xf16> into tensor<786432xf16>
  return %collapsed_9 : tensor<786432xf16>
}
