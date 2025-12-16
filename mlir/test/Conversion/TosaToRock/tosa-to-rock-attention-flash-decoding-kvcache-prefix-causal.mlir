// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o - | FileCheck %s

// CHECK-LABEL: func @mlir_attention
// CHECK: rock.attention
// CHECK: currentSeqLen = ({{.*}} : tensor<8xi32>)
// CHECK-NEXT: prefixOffset = ({{.*}} : tensor<8xi32>)
// CHECK-NEXT: causal
// CHECK-NEXT: lse = {{.*}} : tensor<8x2xf32>

module {
  func.func @mlir_attention(%arg0: tensor<48xf16>, %arg1: tensor<32xf16>, %arg2: tensor<2xi32>, %arg3: tensor<32xf16>, %arg4: tensor<2xi32>) -> (tensor<32xf16>, tensor<16xf32>) attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
    %0 = "tosa.const"() <{values = dense<[[0, 1, 2, 3]]> : tensor<1x4xi32>}> : () -> tensor<1x4xi32>
    %1 = tosa.const_shape  {values = dense<16> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.const_shape  {values = dense<32> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x2x2x2xf32>}> : () -> tensor<2x2x2x2x2xf32>
    %4 = tosa.const_shape  {values = dense<[2, 1, 2, 1, 2]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %5 = "tosa.const"() <{values = dense<1> : tensor<2x2x2x2x2xi8>}> : () -> tensor<2x2x2x2x2xi8>
    %6 = tosa.const_shape  {values = dense<[1, 1, 2, 2, 2]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %8 = tosa.const_shape  {values = dense<[8, 2, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = "tosa.const"() <{values = dense<5.000000e-01> : tensor<2x2x2x2x2xf16>}> : () -> tensor<2x2x2x2x2xf16>
    %10 = "tosa.const"() <{values = dense<0xFC00> : tensor<2x2x2x2x2xf16>}> : () -> tensor<2x2x2x2x2xf16>
    %11 = tosa.const_shape  {values = dense<2> : tensor<5xindex>} : () -> !tosa.shape<5>
    %12 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %13 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x6x2x2x2xf16>}> : () -> tensor<2x6x2x2x2xf16>
    %14 = tosa.const_shape  {values = dense<[2, 6, 1, 2, 2]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %15 = "tosa.const"() <{values = dense<1> : tensor<2x4xi32>}> : () -> tensor<2x4xi32>
    %16 = "tosa.const"() <{values = dense<[[0], [1]]> : tensor<2x1xi32>}> : () -> tensor<2x1xi32>
    %17 = tosa.const_shape  {values = dense<[2, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %expanded = tensor.expand_shape %arg2 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
    %expanded_0 = tensor.expand_shape %arg4 [[0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
    %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 6, 1, 2, 2] : tensor<48xf16> into tensor<2x6x1x2x2xf16>
    %18 = tosa.mul %expanded_1, %13, %12 : (tensor<2x6x1x2x2xf16>, tensor<2x6x2x2x2xf16>, tensor<1xi8>) -> tensor<2x6x2x2x2xf16>
    %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [2, 2, 2, 2, 2] : tensor<32xf16> into tensor<2x2x2x2x2xf16>
    %extracted_slice = tensor.extract_slice %18[0, 0, 0, 0, 0] [2, 2, 2, 2, 2] [1, 1, 1, 1, 1] : tensor<2x6x2x2x2xf16> to tensor<2x2x2x2x2xf16>
    %19 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<2x2x2x2x2xf16>) -> tensor<2x2x2x2x2xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1, 2], [3], [4]] : tensor<2x2x2x2x2xf16> into tensor<8x2x2xf16>
    %collapsed_3 = tensor.collapse_shape %19 [[0, 1, 2], [3], [4]] : tensor<2x2x2x2x2xf16> into tensor<8x2x2xf16>
    %20 = tosa.matmul %collapsed, %collapsed_3, %7, %7 {acc_type = f32} : (tensor<8x2x2xf16>, tensor<8x2x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<8x2x2xf16>
    %expanded_4 = tensor.expand_shape %20 [[0, 1, 2], [3], [4]] output_shape [2, 2, 2, 2, 2] : tensor<8x2x2xf16> into tensor<2x2x2x2x2xf16>
    %21 = tosa.mul %expanded_4, %9, %12 : (tensor<2x2x2x2x2xf16>, tensor<2x2x2x2x2xf16>, tensor<1xi8>) -> tensor<2x2x2x2x2xf16>
    %22 = tosa.add %expanded_0, %16 : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi32>
    %23 = tosa.mul %22, %15, %12 : (tensor<2x1xi32>, tensor<2x4xi32>, tensor<1xi8>) -> tensor<2x4xi32>
    %24 = tosa.mul %0, %15, %12 : (tensor<1x4xi32>, tensor<2x4xi32>, tensor<1xi8>) -> tensor<2x4xi32>
    %25 = tosa.greater %24, %23 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %26 = tosa.cast %25 : (tensor<2x4xi1>) -> tensor<2x4xi32>
    %27 = tosa.cast %26 : (tensor<2x4xi32>) -> tensor<2x4xi8>
    %expanded_5 = tensor.expand_shape %27 [[0, 1, 2], [3, 4]] output_shape [1, 1, 2, 2, 2] : tensor<2x4xi8> into tensor<1x1x2x2x2xi8>
    %28 = tosa.mul %expanded_5, %5, %12 : (tensor<1x1x2x2x2xi8>, tensor<2x2x2x2x2xi8>, tensor<1xi8>) -> tensor<2x2x2x2x2xi8>
    %29 = tosa.cast %28 : (tensor<2x2x2x2x2xi8>) -> tensor<2x2x2x2x2xi1>
    %30 = tosa.select %29, %10, %21 : (tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf16>, tensor<2x2x2x2x2xf16>) -> tensor<2x2x2x2x2xf16>
    %31 = tosa.mul %expanded, %15, %12 : (tensor<2x1xi32>, tensor<2x4xi32>, tensor<1xi8>) -> tensor<2x4xi32>
    %32 = tosa.greater %24, %31 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %33 = tosa.cast %32 : (tensor<2x4xi1>) -> tensor<2x4xi32>
    %34 = tosa.cast %33 : (tensor<2x4xi32>) -> tensor<2x4xi8>
    %expanded_6 = tensor.expand_shape %34 [[0, 1], [2, 3, 4]] output_shape [2, 1, 2, 1, 2] : tensor<2x4xi8> into tensor<2x1x2x1x2xi8>
    %35 = tosa.mul %expanded_6, %5, %12 : (tensor<2x1x2x1x2xi8>, tensor<2x2x2x2x2xi8>, tensor<1xi8>) -> tensor<2x2x2x2x2xi8>
    %36 = tosa.cast %35 : (tensor<2x2x2x2x2xi8>) -> tensor<2x2x2x2x2xi1>
    %37 = tosa.select %36, %10, %30 : (tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf16>, tensor<2x2x2x2x2xf16>) -> tensor<2x2x2x2x2xf16>
    %38 = tosa.cast %37 : (tensor<2x2x2x2x2xf16>) -> tensor<2x2x2x2x2xf32>
    %39 = tosa.reduce_max %38 {axis = 4 : i32} : (tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x1xf32>
    %40 = tosa.mul %39, %3, %12 : (tensor<2x2x2x2x1xf32>, tensor<2x2x2x2x2xf32>, tensor<1xi8>) -> tensor<2x2x2x2x2xf32>
    %41 = tosa.sub %38, %40 : (tensor<2x2x2x2x2xf32>, tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>
    %42 = tosa.exp %41 : (tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>
    %43 = tosa.reduce_sum %42 {axis = 4 : i32} : (tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x1xf32>
    %44 = tosa.mul %43, %3, %12 : (tensor<2x2x2x2x1xf32>, tensor<2x2x2x2x2xf32>, tensor<1xi8>) -> tensor<2x2x2x2x2xf32>
    %45 = tosa.reciprocal %44 : (tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>
    %46 = tosa.mul %42, %45, %12 : (tensor<2x2x2x2x2xf32>, tensor<2x2x2x2x2xf32>, tensor<1xi8>) -> tensor<2x2x2x2x2xf32>
    %47 = tosa.cast %46 : (tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf16>
    %collapsed_7 = tensor.collapse_shape %47 [[0, 1, 2], [3], [4]] : tensor<2x2x2x2x2xf16> into tensor<8x2x2xf16>
    %expanded_8 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [8, 2, 2] : tensor<32xf16> into tensor<8x2x2xf16>
    %48 = tosa.matmul %collapsed_7, %expanded_8, %7, %7 {acc_type = f32} : (tensor<8x2x2xf16>, tensor<8x2x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<8x2x2xf16>
    %expanded_9 = tensor.expand_shape %48 [[0, 1, 2], [3], [4]] output_shape [2, 2, 2, 2, 2] : tensor<8x2x2xf16> into tensor<2x2x2x2x2xf16>
    %49 = tosa.transpose %expanded_9 {perms = array<i32: 0, 2, 3, 1, 4>} : (tensor<2x2x2x2x2xf16>) -> tensor<2x2x2x2x2xf16>
    %collapsed_10 = tensor.collapse_shape %49 [[0, 1, 2, 3, 4]] : tensor<2x2x2x2x2xf16> into tensor<32xf16>
    %50 = tosa.log %43 : (tensor<2x2x2x2x1xf32>) -> tensor<2x2x2x2x1xf32>
    %51 = tosa.add %39, %50 : (tensor<2x2x2x2x1xf32>, tensor<2x2x2x2x1xf32>) -> tensor<2x2x2x2x1xf32>
    %collapsed_11 = tensor.collapse_shape %51 [[0, 1, 2, 3, 4]] : tensor<2x2x2x2x1xf32> into tensor<16xf32>
    return %collapsed_10, %collapsed_11 : tensor<32xf16>, tensor<16xf32>
  }
}