// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @test_deref_basic
func.func @test_deref_basic(
    %arg0: tensor<1x64x1xi64> {mhal.read_access}
) -> tensor<1x64x8192xf16> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<1> : tensor<1x64x8192xi64>}> : () -> tensor<1x64x8192xi64>
  %1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 = tosa.mul %arg0, %0, %1 : (tensor<1x64x1xi64>, tensor<1x64x8192xi64>, tensor<1xi8>) -> tensor<1x64x8192xi64>
  %3 = "tosa.const"() <{values = dense<1> : tensor<1x64x8192xi64>}> : () -> tensor<1x64x8192xi64>
  %4 = tosa.mul %3, %0, %1 : (tensor<1x64x8192xi64>, tensor<1x64x8192xi64>, tensor<1xi8>) -> tensor<1x64x8192xi64>
  %5 = tosa.add %2, %4 : (tensor<1x64x8192xi64>, tensor<1x64x8192xi64>) -> tensor<1x64x8192xi64>
  %6 = tosa.custom %5 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "deref"} : (tensor<1x64x8192xi64>) -> tensor<1x64x8192xf16>
  // CHECK: rock.deref
  // CHECK-SAME: tensor<1x64x1xi64> -> tensor<1x64x8192xf16>
  return %6 : tensor<1x64x8192xf16>
}

// CHECK-LABEL: func @test_paged_attention
func.func @test_paged_attention(
    %arg0: tensor<1024xi64> {mhal.read_access},
    %arg1: tensor<1024xi64> {mhal.read_access},
    %arg2: tensor<1xi32> {mhal.read_access},
    %arg3: tensor<1728000xf16> {mhal.read_access}
) -> tensor<1344000xf16> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<1x14x1500x4096xf16>}> : () -> tensor<1x14x1500x4096xf16>
  %1 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x14x1500x4096xf16>}> : () -> tensor<1x14x1500x4096xf16>
  %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %4 = "tosa.const"() <{values = dense<1> : tensor<1x64x8192xi64>}> : () -> tensor<1x64x8192xi64>
  %expanded = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 1500, 18, 64] : tensor<1728000xf16> into tensor<1x1500x18x64xf16>
  %5 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x1500x18x64xf16>) -> tensor<1x18x1500x64xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xi64> into tensor<1x16x64xi64>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xi64> into tensor<1x16x64xi64>
  %extracted_slice = tensor.extract_slice %expanded_1[0, 0, 0] [1, 1, 64] [1, 1, 1] : tensor<1x16x64xi64> to tensor<1x1x64xi64>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x1x64xi64> into tensor<1x64xi64>
  %expanded_2 = tensor.expand_shape %collapsed [[0], [1, 2]] output_shape [1, 64, 1] : tensor<1x64xi64> into tensor<1x64x1xi64>
  %6 = tosa.mul %expanded_2, %4, %2 : (tensor<1x64x1xi64>, tensor<1x64x8192xi64>, tensor<1xi8>) -> tensor<1x64x8192xi64>
  %7 = tosa.mul %4, %4, %2 : (tensor<1x64x8192xi64>, tensor<1x64x8192xi64>, tensor<1xi8>) -> tensor<1x64x8192xi64>
  %8 = tosa.add %6, %7 : (tensor<1x64x8192xi64>, tensor<1x64x8192xi64>) -> tensor<1x64x8192xi64>

  // CHECK: %[[VAL_DEREF:.*]] = rock.deref %{{.*}} : tensor<1x64x1xi64> -> tensor<1x64x8192xf16>
  %9 = tosa.custom %8 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "deref"} : (tensor<1x64x8192xi64>) -> tensor<1x64x8192xf16>
  %extracted_slice_3 = tensor.extract_slice %expanded_0[0, 0, 0] [1, 1, 64] [1, 1, 1] : tensor<1x16x64xi64> to tensor<1x1x64xi64>
  %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [[0, 1], [2]] : tensor<1x1x64xi64> into tensor<1x64xi64>
  %expanded_5 = tensor.expand_shape %collapsed_4 [[0], [1, 2]] output_shape [1, 64, 1] : tensor<1x64xi64> into tensor<1x64x1xi64>
  %10 = tosa.mul %expanded_5, %4, %2 : (tensor<1x64x1xi64>, tensor<1x64x8192xi64>, tensor<1xi8>) -> tensor<1x64x8192xi64>
  %11 = tosa.add %10, %7 : (tensor<1x64x8192xi64>, tensor<1x64x8192xi64>) -> tensor<1x64x8192xi64>

  // CHECK: %[[KEY_DEREF:.*]] = rock.deref %{{.*}} : tensor<1x64x1xi64> -> tensor<1x64x8192xf16>
  %12 = tosa.custom %11 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "deref"} : (tensor<1x64x8192xi64>) -> tensor<1x64x8192xf16>
  %extracted_slice_6 = tensor.extract_slice %5[0, 0, 0, 0] [1, 14, 1500, 64] [1, 1, 1, 1] : tensor<1x18x1500x64xf16> to tensor<1x14x1500x64xf16>
  %collapsed_7 = tensor.collapse_shape %9 [[0], [1, 2]] : tensor<1x64x8192xf16> into tensor<1x524288xf16>
  %expanded_8 = tensor.expand_shape %collapsed_7 [[0], [1, 2, 3, 4]] output_shape [1, 2, 1, 4096, 64] : tensor<1x524288xf16> into tensor<1x2x1x4096x64xf16>
  %collapsed_9 = tensor.collapse_shape %12 [[0], [1, 2]] : tensor<1x64x8192xf16> into tensor<1x524288xf16>
  %expanded_10 = tensor.expand_shape %collapsed_9 [[0], [1, 2, 3, 4]] output_shape [1, 2, 1, 4096, 64] : tensor<1x524288xf16> into tensor<1x2x1x4096x64xf16>
  %13 = tosa.transpose %expanded_10 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<1x2x1x4096x64xf16>) -> tensor<1x2x1x64x4096xf16>
  %14 = "tosa.const"() <{values = dense<1.> : tensor<1x2x7x4096x64xf16>}> : () -> tensor<1x2x7x4096x64xf16>
  %15 = tosa.mul %expanded_8, %14, %2 : (tensor<1x2x1x4096x64xf16>, tensor<1x2x7x4096x64xf16>, tensor<1xi8>) -> tensor<1x2x7x4096x64xf16>
  %16 = "tosa.const"() <{values = dense<1.> : tensor<1x2x7x64x4096xf16>}> : () -> tensor<1x2x7x64x4096xf16>
  %17 = tosa.mul %13, %16, %2 : (tensor<1x2x1x64x4096xf16>, tensor<1x2x7x64x4096xf16>, tensor<1xi8>) -> tensor<1x2x7x64x4096xf16>
  %collapsed_11 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2], [3]] : tensor<1x14x1500x64xf16> into tensor<14x1500x64xf16>
  %collapsed_12 = tensor.collapse_shape %17 [[0, 1, 2], [3], [4]] : tensor<1x2x7x64x4096xf16> into tensor<14x64x4096xf16>
  %collapsed_13 = tensor.collapse_shape %15 [[0, 1, 2], [3], [4]] : tensor<1x2x7x4096x64xf16> into tensor<14x4096x64xf16>
  %18 = tosa.matmul %collapsed_11, %collapsed_12, %3, %3 {acc_type = f32} : (tensor<14x1500x64xf16>, tensor<14x64x4096xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x1500x4096xf16>
  %expanded_14 = tensor.expand_shape %18 [[0, 1], [2], [3]] output_shape [1, 14, 1500, 4096] : tensor<14x1500x4096xf16> into tensor<1x14x1500x4096xf16>
  %19 = tosa.mul %expanded_14, %0, %2 : (tensor<1x14x1500x4096xf16>, tensor<1x14x1500x4096xf16>, tensor<1xi8>) -> tensor<1x14x1500x4096xf16>
  %cst_shape_4d = tosa.const_shape {values = dense<[1, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %20 = tosa.reshape %arg2, %cst_shape_4d : (tensor<1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
  %21 = "tosa.const"() <{values = dense<1> : tensor<1x14x1500x4096xi32>}> : () -> tensor<1x14x1500x4096xi32>
  %22 = tosa.mul %20, %21, %2 : (tensor<1x1x1x1xi32>, tensor<1x14x1500x4096xi32>, tensor<1xi8>) -> tensor<1x14x1500x4096xi32>

  // CHECK: rock.attention
  // CHECK: keyAddresses = (%[[KEY_DEREF]] : tensor<1x64x8192xf16>)
  // CHECK: valueAddresses = (%[[VAL_DEREF]] : tensor<1x64x8192xf16>)

  %23 = "tosa.const"() <{values = dense<0> : tensor<1x14x1500x4096xi32>}> : () -> tensor<1x14x1500x4096xi32>
  %24 = tosa.greater %23, %22 : (tensor<1x14x1500x4096xi32>, tensor<1x14x1500x4096xi32>) -> tensor<1x14x1500x4096xi1>
  %25 = tosa.select %24, %1, %19 : (tensor<1x14x1500x4096xi1>, tensor<1x14x1500x4096xf16>, tensor<1x14x1500x4096xf16>) -> tensor<1x14x1500x4096xf16>
  %26 = tosa.reduce_max %25 {axis = 3 : i32} : (tensor<1x14x1500x4096xf16>) -> tensor<1x14x1500x1xf16>
  %27 = tosa.sub %25, %26 : (tensor<1x14x1500x4096xf16>, tensor<1x14x1500x1xf16>) -> tensor<1x14x1500x4096xf16>
  %28 = tosa.exp %27 : (tensor<1x14x1500x4096xf16>) -> tensor<1x14x1500x4096xf16>
  %29 = tosa.reduce_sum %28 {axis = 3 : i32} : (tensor<1x14x1500x4096xf16>) -> tensor<1x14x1500x1xf16>
  %30 = tosa.reciprocal %29 : (tensor<1x14x1500x1xf16>) -> tensor<1x14x1500x1xf16>
  %31 = tosa.mul %28, %30, %2 : (tensor<1x14x1500x4096xf16>, tensor<1x14x1500x1xf16>, tensor<1xi8>) -> tensor<1x14x1500x4096xf16>
  %collapsed_16 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x14x1500x4096xf16> into tensor<14x1500x4096xf16>
  %32 = tosa.matmul %collapsed_16, %collapsed_13, %3, %3 {acc_type = f32} : (tensor<14x1500x4096xf16>, tensor<14x4096x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x1500x64xf16>
  %expanded_17 = tensor.expand_shape %32 [[0, 1], [2], [3]] output_shape [1, 14, 1500, 64] : tensor<14x1500x64xf16> into tensor<1x14x1500x64xf16>
  %33 = tosa.transpose %expanded_17 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x14x1500x64xf16>) -> tensor<1x1500x14x64xf16>
  %collapsed_18 = tensor.collapse_shape %33 [[0, 1, 2, 3]] : tensor<1x1500x14x64xf16> into tensor<1344000xf16>
  return %collapsed_18 : tensor<1344000xf16>
}
