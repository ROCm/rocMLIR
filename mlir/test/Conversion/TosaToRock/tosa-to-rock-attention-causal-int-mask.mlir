// This test checks that we do not match an integer mask as causal when
// pattern matching for attention in TosaToRock

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o -| FileCheck %s


// CHECK-LABEL: func @mlir_causal_attention_no_select_int_mask
// CHECK: rock.attention
// CHECK: qk = {{.*}} * {{.*}} : tensor<14x8x64xf16>, tensor<14x64x8xf16>
// CHECK-NOT: causal
func.func @mlir_causal_attention_no_select_int_mask(%arg0: tensor<1024xf16>, %arg1: tensor<7168xf16>, %arg2: tensor<7168xf16>) -> tensor<7168xf16> attributes {kernel} {
  %0 = tosa.const_shape  {values = dense<7168> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.const_shape  {values = dense<[14, 8, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x14x8x8xf32>}> : () -> tensor<1x14x8x8xf32>
  %3 = tosa.const_shape  {values = dense<[1, 14, 8, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %5 = tosa.const_shape  {values = dense<[14, 64, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %6 = tosa.const_shape  {values = dense<[14, 8, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %7 = "tosa.const"() <{values = dense<1> : tensor<1x14x8x8xi16>}> : () -> tensor<1x14x8x8xi16>
  %8 = "tosa.const"() <{values = dense<3.535160e-01> : tensor<1x14x64x8xf16>}> : () -> tensor<1x14x64x8xf16>
  %9 = tosa.const_shape  {values = dense<[1, 14, 64, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %10 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %11 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x7x64x8xf16>}> : () -> tensor<1x2x7x64x8xf16>
  %12 = tosa.const_shape  {values = dense<[1, 2, 1, 8, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %13 = "tosa.const"() <{values = dense<[[[[0, -32768, -32768, -32768, -32768, -32768, -32768, -32768], [0, 0, -32768, -32768, -32768, -32768, -32768, -32768], [0, 0, 0, -32768, -32768, -32768, -32768, -32768], [0, 0, 0, 0, -32768, -32768, -32768, -32768], [0, 0, 0, 0, 0, -32768, -32768, -32768], [0, 0, 0, 0, 0, 0, -32768, -32768], [0, 0, 0, 0, 0, 0, 0, -32768], [0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x8x8xi16>}> : () -> tensor<1x1x8x8xi16>
  %14 = tosa.const_shape  {values = dense<[1, 8, 14, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 8, 14, 64] : tensor<7168xf16> into tensor<1x8x14x64xf16>
  %15 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x8x14x64xf16>) -> tensor<1x14x8x64xf16>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 2, 1, 8, 64] : tensor<1024xf16> into tensor<1x2x1x8x64xf16>
  %16 = tosa.transpose %expanded_0 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<1x2x1x8x64xf16>) -> tensor<1x2x1x64x8xf16>
  %17 = tosa.mul %16, %11, %10 : (tensor<1x2x1x64x8xf16>, tensor<1x2x7x64x8xf16>, tensor<1xi8>) -> tensor<1x2x7x64x8xf16>
  %collapsed = tensor.collapse_shape %17 [[0], [1, 2], [3], [4]] : tensor<1x2x7x64x8xf16> into tensor<1x14x64x8xf16>
  %18 = tosa.mul %collapsed, %8, %10 : (tensor<1x14x64x8xf16>, tensor<1x14x64x8xf16>, tensor<1xi8>) -> tensor<1x14x64x8xf16>
  %19 = tosa.mul %13, %7, %10 : (tensor<1x1x8x8xi16>, tensor<1x14x8x8xi16>, tensor<1xi8>) -> tensor<1x14x8x8xi16>
  %collapsed_1 = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<1x14x8x64xf16> into tensor<14x8x64xf16>
  %collapsed_2 = tensor.collapse_shape %18 [[0, 1], [2], [3]] : tensor<1x14x64x8xf16> into tensor<14x64x8xf16>
  %20 = tosa.matmul %collapsed_1, %collapsed_2, %4, %4 {acc_type = f32} : (tensor<14x8x64xf16>, tensor<14x64x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x8x8xf16>
  %expanded_3 = tensor.expand_shape %20 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf16> into tensor<1x14x8x8xf16>
  %21 = tosa.cast %expanded_3 : (tensor<1x14x8x8xf16>) -> tensor<1x14x8x8xi16>
  %22 = tosa.add %21, %19 : (tensor<1x14x8x8xi16>, tensor<1x14x8x8xi16>) -> tensor<1x14x8x8xi16>
  %23 = tosa.cast %22 : (tensor<1x14x8x8xi16>) -> tensor<1x14x8x8xf16>
  %24 = tosa.cast %23 : (tensor<1x14x8x8xf16>) -> tensor<1x14x8x8xf32>
  %25 = tosa.reduce_max %24 {axis = 3 : i32} : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x1xf32>
  %26 = tosa.mul %25, %2, %10 : (tensor<1x14x8x1xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %27 = tosa.sub %24, %26 : (tensor<1x14x8x8xf32>, tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %28 = tosa.exp %27 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %29 = tosa.reduce_sum %28 {axis = 3 : i32} : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x1xf32>
  %30 = tosa.mul %29, %2, %10 : (tensor<1x14x8x1xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %31 = tosa.reciprocal %30 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %32 = tosa.mul %28, %31, %10 : (tensor<1x14x8x8xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %33 = tosa.cast %32 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf16>
  %collapsed_4 = tensor.collapse_shape %33 [[0, 1], [2], [3]] : tensor<1x14x8x8xf16> into tensor<14x8x8xf16>
  %expanded_5 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [14, 8, 64] : tensor<7168xf16> into tensor<14x8x64xf16>
  %34 = tosa.matmul %collapsed_4, %expanded_5, %4, %4 {acc_type = f32} : (tensor<14x8x8xf16>, tensor<14x8x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x8x64xf16>
  %collapsed_6 = tensor.collapse_shape %34 [[0, 1, 2]] : tensor<14x8x64xf16> into tensor<7168xf16>
  return %collapsed_6 : tensor<7168xf16>
}