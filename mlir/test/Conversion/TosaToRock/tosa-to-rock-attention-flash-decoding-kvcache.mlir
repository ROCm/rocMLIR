// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o - | FileCheck %s

// CHECK-LABEL: func @mlir_attention
// CHECK: rock.attention
// CHECK: currentSeqLen = ({{.*}} : tensor<8xi32>)
// CHECK: lse = {{.*}} : tensor<8x1xf32>

module {
  func.func @mlir_attention(%arg0: tensor<24xf16>, %arg1: tensor<32xf16>, %arg2: tensor<2xi32>, %arg3: tensor<32xf16>) -> (tensor<16xf16>, tensor<8xf32>) attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
    %0 = "tosa.const"() <{values = dense<[[0, 1, 2, 3]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
    %1 = tosa.const_shape  {values = dense<8> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.const_shape  {values = dense<16> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x2x1x2xf32>}> : () -> tensor<2x2x2x1x2xf32>
    %4 = "tosa.const"() <{values = dense<1> : tensor<2x2x2x1x2xi8>}> : () -> tensor<2x2x2x1x2xi8>
    %5 = tosa.const_shape  {values = dense<[2, 1, 2, 1, 2]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %6 = tosa.const_shape  {values = dense<2> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = "tosa.const"() <{values = dense<1> : tensor<2x4xi32>}> : () -> tensor<2x4xi32>
    %8 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %9 = tosa.const_shape  {values = dense<[8, 2, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10 = tosa.const_shape  {values = dense<[8, 1, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %11 = "tosa.const"() <{values = dense<5.000000e-01> : tensor<2x2x2x1x2xf16>}> : () -> tensor<2x2x2x1x2xf16>
    %12 = "tosa.const"() <{values = dense<0xFC00> : tensor<2x2x2x1x2xf16>}> : () -> tensor<2x2x2x1x2xf16>
    %13 = tosa.const_shape  {values = dense<[2, 2, 2, 1, 2]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %14 = tosa.const_shape  {values = dense<2> : tensor<5xindex>} : () -> !tosa.shape<5>
    %15 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %16 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x6x2x1x2xf16>}> : () -> tensor<2x6x2x1x2xf16>
    %17 = tosa.const_shape  {values = dense<[2, 6, 1, 1, 2]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %18 = tosa.const_shape  {values = dense<[2, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %expanded = tensor.expand_shape %arg2 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 6, 1, 1, 2] : tensor<24xf16> into tensor<2x6x1x1x2xf16>
    %19 = tosa.mul %expanded_0, %16, %15 : (tensor<2x6x1x1x2xf16>, tensor<2x6x2x1x2xf16>, tensor<1xi8>) -> tensor<2x6x2x1x2xf16>
    %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [2, 2, 2, 2, 2] : tensor<32xf16> into tensor<2x2x2x2x2xf16>
    %extracted_slice = tensor.extract_slice %19[0, 0, 0, 0, 0] [2, 2, 2, 1, 2] [1, 1, 1, 1, 1] : tensor<2x6x2x1x2xf16> to tensor<2x2x2x1x2xf16>
    %20 = tosa.transpose %expanded_1 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<2x2x2x2x2xf16>) -> tensor<2x2x2x2x2xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1, 2], [3], [4]] : tensor<2x2x2x1x2xf16> into tensor<8x1x2xf16>
    %collapsed_2 = tensor.collapse_shape %20 [[0, 1, 2], [3], [4]] : tensor<2x2x2x2x2xf16> into tensor<8x2x2xf16>
    %21 = tosa.matmul %collapsed, %collapsed_2, %8, %8 {acc_type = f32} : (tensor<8x1x2xf16>, tensor<8x2x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<8x1x2xf16>
    %expanded_3 = tensor.expand_shape %21 [[0, 1, 2], [3], [4]] output_shape [2, 2, 2, 1, 2] : tensor<8x1x2xf16> into tensor<2x2x2x1x2xf16>
    %22 = tosa.mul %expanded_3, %11, %15 : (tensor<2x2x2x1x2xf16>, tensor<2x2x2x1x2xf16>, tensor<1xi8>) -> tensor<2x2x2x1x2xf16>
    %23 = tosa.mul %0, %7, %15 : (tensor<1x4xi32>, tensor<2x4xi32>, tensor<1xi8>) -> tensor<2x4xi32>
    %expanded_4 = tensor.expand_shape %23 [[0], [1, 2]] output_shape [2, 2, 2] : tensor<2x4xi32> into tensor<2x2x2xi32>
    %24 = tosa.mul %expanded, %7, %15 : (tensor<2x1xi32>, tensor<2x4xi32>, tensor<1xi8>) -> tensor<2x4xi32>
    %expanded_5 = tensor.expand_shape %24 [[0], [1, 2]] output_shape [2, 2, 2] : tensor<2x4xi32> into tensor<2x2x2xi32>
    %25 = tosa.greater %expanded_4, %expanded_5 : (tensor<2x2x2xi32>, tensor<2x2x2xi32>) -> tensor<2x2x2xi1>
    %26 = tosa.cast %25 : (tensor<2x2x2xi1>) -> tensor<2x2x2xi32>
    %27 = tosa.cast %26 : (tensor<2x2x2xi32>) -> tensor<2x2x2xi8>
    %expanded_6 = tensor.expand_shape %27 [[0, 1], [2, 3], [4]] output_shape [2, 1, 2, 1, 2] : tensor<2x2x2xi8> into tensor<2x1x2x1x2xi8>
    %28 = tosa.mul %expanded_6, %4, %15 : (tensor<2x1x2x1x2xi8>, tensor<2x2x2x1x2xi8>, tensor<1xi8>) -> tensor<2x2x2x1x2xi8>
    %29 = tosa.cast %28 : (tensor<2x2x2x1x2xi8>) -> tensor<2x2x2x1x2xi1>
    %30 = tosa.select %29, %12, %22 : (tensor<2x2x2x1x2xi1>, tensor<2x2x2x1x2xf16>, tensor<2x2x2x1x2xf16>) -> tensor<2x2x2x1x2xf16>
    %31 = tosa.cast %30 : (tensor<2x2x2x1x2xf16>) -> tensor<2x2x2x1x2xf32>
    %32 = tosa.reduce_max %31 {axis = 4 : i32} : (tensor<2x2x2x1x2xf32>) -> tensor<2x2x2x1x1xf32>
    %33 = tosa.mul %32, %3, %15 : (tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x2xf32>, tensor<1xi8>) -> tensor<2x2x2x1x2xf32>
    %34 = tosa.sub %31, %33 : (tensor<2x2x2x1x2xf32>, tensor<2x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>
    %35 = tosa.exp %34 : (tensor<2x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>
    %36 = tosa.reduce_sum %35 {axis = 4 : i32} : (tensor<2x2x2x1x2xf32>) -> tensor<2x2x2x1x1xf32>
    %37 = tosa.mul %36, %3, %15 : (tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x2xf32>, tensor<1xi8>) -> tensor<2x2x2x1x2xf32>
    %38 = tosa.reciprocal %37 : (tensor<2x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>
    %39 = tosa.mul %35, %38, %15 : (tensor<2x2x2x1x2xf32>, tensor<2x2x2x1x2xf32>, tensor<1xi8>) -> tensor<2x2x2x1x2xf32>
    %40 = tosa.cast %39 : (tensor<2x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf16>
    %collapsed_7 = tensor.collapse_shape %40 [[0, 1, 2], [3], [4]] : tensor<2x2x2x1x2xf16> into tensor<8x1x2xf16>
    %expanded_8 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [8, 2, 2] : tensor<32xf16> into tensor<8x2x2xf16>
    %41 = tosa.matmul %collapsed_7, %expanded_8, %8, %8 {acc_type = f32} : (tensor<8x1x2xf16>, tensor<8x2x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<8x1x2xf16>
    %expanded_9 = tensor.expand_shape %41 [[0, 1, 2], [3], [4]] output_shape [2, 2, 2, 1, 2] : tensor<8x1x2xf16> into tensor<2x2x2x1x2xf16>
    %42 = tosa.transpose %expanded_9 {perms = array<i32: 0, 2, 3, 1, 4>} : (tensor<2x2x2x1x2xf16>) -> tensor<2x2x1x2x2xf16>
    %collapsed_10 = tensor.collapse_shape %42 [[0, 1, 2, 3, 4]] : tensor<2x2x1x2x2xf16> into tensor<16xf16>
    %43 = tosa.log %36 : (tensor<2x2x2x1x1xf32>) -> tensor<2x2x2x1x1xf32>
    %44 = tosa.add %32, %43 : (tensor<2x2x2x1x1xf32>, tensor<2x2x2x1x1xf32>) -> tensor<2x2x2x1x1xf32>
    %collapsed_11 = tensor.collapse_shape %44 [[0, 1, 2, 3, 4]] : tensor<2x2x2x1x1xf32> into tensor<8xf32>
    return %collapsed_10, %collapsed_11 : tensor<16xf16>, tensor<8xf32>
  }
}