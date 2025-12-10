// RUN: rocmlir-opt --tosa-to-rock %s | FileCheck %s

module {
  // CHECK-LABEL: func @mlir_attention
  // CHECK: rock.attention
  // CHECK: currentSeqLen = (%{{.*}} : tensor<14xi32>)
  // CHECK-NOT: causal
  // CHECK: prefixCausal
  func.func @mlir_attention(%arg0: tensor<1xi32>, %arg1: tensor<4608xf16>, %arg2: tensor<2048xf16>, %arg3: tensor<14336xf16>) -> tensor<3584xf16> attributes {kernel} {
    %0 = "tosa.const"() <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<1x16xi32>}> : () -> tensor<1x16xi32>
    %1 = tosa.const_shape  {values = dense<3584> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.const_shape  {values = dense<[14, 16, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.const_shape  {values = dense<[14, 4, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x14x4x16xf32>}> : () -> tensor<1x14x4x16xf32>
    %5 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x14x4x16xf16>}> : () -> tensor<1x14x4x16xf16>
    %6 = tosa.const_shape  {values = dense<[1, 14, 4, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %8 = tosa.const_shape  {values = dense<[14, 64, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.const_shape  {values = dense<[14, 4, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x7x64x16xf16>}> : () -> tensor<1x2x7x64x16xf16>
    %11 = tosa.const_shape  {values = dense<[1, 14, 4, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %12 = "tosa.const"() <{values = dense<1> : tensor<1x14x4x16xi8>}> : () -> tensor<1x14x4x16xi8>
    %13 = tosa.const_shape  {values = dense<[1, 1, 4, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %14 = "tosa.const"() <{values = dense<1> : tensor<4x1xi32>}> : () -> tensor<4x1xi32>
    %15 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<1x14x4x16xf16>}> : () -> tensor<1x14x4x16xf16>
    %16 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %17 = "tosa.const"() <{values = dense<1> : tensor<4x16xi32>}> : () -> tensor<4x16xi32>
    %18 = "tosa.const"() <{values = dense<[[0], [1], [2], [3]]> : tensor<4x1xi32>}> : () -> tensor<4x1xi32>
    %19 = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
    %20 = tosa.const_shape  {values = dense<[1, 4, 18, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %21 = tosa.const_shape  {values = dense<[1, 2, 1, 16, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [1, 2, 1, 16, 64] : tensor<2048xf16> into tensor<1x2x1x16x64xf16>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 4, 18, 64] : tensor<4608xf16> into tensor<1x4x18x64xf16>
    %22 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x4x18x64xf16>) -> tensor<1x18x4x64xf16>
    %expanded_1 = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
    %23 = tosa.mul %0, %17, %16 : (tensor<1x16xi32>, tensor<4x16xi32>, tensor<1xi8>) -> tensor<4x16xi32>
    %24 = tosa.mul %expanded_1, %14, %16 : (tensor<1x1xi32>, tensor<4x1xi32>, tensor<1xi8>) -> tensor<4x1xi32>
    %25 = tosa.add %24, %18 : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x1xi32>
    %26 = tosa.mul %25, %17, %16 : (tensor<4x1xi32>, tensor<4x16xi32>, tensor<1xi8>) -> tensor<4x16xi32>
    %27 = tosa.greater %23, %26 : (tensor<4x16xi32>, tensor<4x16xi32>) -> tensor<4x16xi1>
    %28 = tosa.cast %27 : (tensor<4x16xi1>) -> tensor<4x16xi32>
    %29 = tosa.cast %28 : (tensor<4x16xi32>) -> tensor<4x16xi8>
    %expanded_2 = tensor.expand_shape %29 [[0, 1, 2], [3]] output_shape [1, 1, 4, 16] : tensor<4x16xi8> into tensor<1x1x4x16xi8>
    %30 = tosa.mul %expanded_2, %12, %16 : (tensor<1x1x4x16xi8>, tensor<1x14x4x16xi8>, tensor<1xi8>) -> tensor<1x14x4x16xi8>
    %extracted_slice = tensor.extract_slice %22[0, 0, 0, 0] [1, 14, 4, 64] [1, 1, 1, 1] : tensor<1x18x4x64xf16> to tensor<1x14x4x64xf16>
    %31 = tosa.transpose %expanded {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<1x2x1x16x64xf16>) -> tensor<1x2x1x64x16xf16>
    %32 = tosa.mul %31, %10, %16 : (tensor<1x2x1x64x16xf16>, tensor<1x2x7x64x16xf16>, tensor<1xi8>) -> tensor<1x2x7x64x16xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x14x4x64xf16> into tensor<14x4x64xf16>
    %collapsed_3 = tensor.collapse_shape %32 [[0, 1, 2], [3], [4]] : tensor<1x2x7x64x16xf16> into tensor<14x64x16xf16>
    %33 = tosa.matmul %collapsed, %collapsed_3, %7, %7 {acc_type = f32} : (tensor<14x4x64xf16>, tensor<14x64x16xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x4x16xf16>
    %expanded_4 = tensor.expand_shape %33 [[0, 1], [2], [3]] output_shape [1, 14, 4, 16] : tensor<14x4x16xf16> into tensor<1x14x4x16xf16>
    %34 = tosa.mul %expanded_4, %15, %16 : (tensor<1x14x4x16xf16>, tensor<1x14x4x16xf16>, tensor<1xi8>) -> tensor<1x14x4x16xf16>
    %35 = tosa.cast %30 : (tensor<1x14x4x16xi8>) -> tensor<1x14x4x16xi1>
    %36 = tosa.select %35, %5, %34 : (tensor<1x14x4x16xi1>, tensor<1x14x4x16xf16>, tensor<1x14x4x16xf16>) -> tensor<1x14x4x16xf16>
    %37 = tosa.cast %36 : (tensor<1x14x4x16xf16>) -> tensor<1x14x4x16xf32>
    %38 = tosa.reduce_max %37 {axis = 3 : i32} : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x1xf32>
    %39 = tosa.mul %38, %4, %16 : (tensor<1x14x4x1xf32>, tensor<1x14x4x16xf32>, tensor<1xi8>) -> tensor<1x14x4x16xf32>
    %40 = tosa.sub %37, %39 : (tensor<1x14x4x16xf32>, tensor<1x14x4x16xf32>) -> tensor<1x14x4x16xf32>
    %41 = tosa.exp %40 : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x16xf32>
    %42 = tosa.reduce_sum %41 {axis = 3 : i32} : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x1xf32>
    %43 = tosa.mul %42, %4, %16 : (tensor<1x14x4x1xf32>, tensor<1x14x4x16xf32>, tensor<1xi8>) -> tensor<1x14x4x16xf32>
    %44 = tosa.reciprocal %43 : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x16xf32>
    %45 = tosa.mul %41, %44, %16 : (tensor<1x14x4x16xf32>, tensor<1x14x4x16xf32>, tensor<1xi8>) -> tensor<1x14x4x16xf32>
    %46 = tosa.cast %45 : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x16xf16>
    %collapsed_5 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<1x14x4x16xf16> into tensor<14x4x16xf16>
    %expanded_6 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [14, 16, 64] : tensor<14336xf16> into tensor<14x16x64xf16>
    %47 = tosa.matmul %collapsed_5, %expanded_6, %7, %7 {acc_type = f32} : (tensor<14x4x16xf16>, tensor<14x16x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x4x64xf16>
    %expanded_7 = tensor.expand_shape %47 [[0, 1], [2], [3]] output_shape [1, 14, 4, 64] : tensor<14x4x64xf16> into tensor<1x14x4x64xf16>
    %48 = tosa.transpose %expanded_7 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x14x4x64xf16>) -> tensor<1x4x14x64xf16>
    %collapsed_8 = tensor.collapse_shape %48 [[0, 1, 2, 3]] : tensor<1x4x14x64xf16> into tensor<3584xf16>
    return %collapsed_8 : tensor<3584xf16>
  }
}