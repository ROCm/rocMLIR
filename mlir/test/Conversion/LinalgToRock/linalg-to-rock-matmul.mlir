// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -verify-diagnostics --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matmul_3D
// CHECK-NEXT: %[[expanded:.*]] = tensor.expand_shape
// CHECK-NEXT: %[[expanded_0:.*]] = tensor.expand_shape
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = %[[expanded_0]] * %[[expanded]] storeMethod =  set
// CHECK-NEXT: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-NEXT: return %[[collapsed]]
func.func @matmul_3D(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<9xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 2, 3] : tensor<6xf32> into tensor<1x2x3xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 3, 2] : tensor<6xf32> into tensor<1x3x2xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<1x3x3xf32>
  %0 = linalg.batch_matmul ins(%expanded_0, %expanded : tensor<1x3x2xf32>, tensor<1x2x3xf32>) outs(%cst : tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<1x3x3xf32> into tensor<9xf32>
  return %collapsed : tensor<9xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_2D
// CHECK-NEXT: %[[expanded:.*]] = tensor.expand_shape
// CHECK-NEXT: %[[expanded_0:.*]] = tensor.expand_shape
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = %[[expanded_0]] * %[[expanded]] storeMethod =  set
// CHECK-NEXT: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-NEXT: return %[[collapsed]]
func.func @matmul_2D(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<9xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [3, 2] : tensor<6xf32> into tensor<3x2xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
  %0 = linalg.matmul ins(%expanded_0, %expanded : tensor<3x2xf32>, tensor<2x3xf32>) outs(%cst : tensor<3x3xf32>) -> tensor<3x3xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<3x3xf32> into tensor<9xf32>
  return %collapsed : tensor<9xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_transposed_A_2D(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = tr %[[arg0]] * %[[arg1]] storeMethod =  set
func.func @matmul_transposed_A_2D(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<3x3xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
  %0 = linalg.matmul
    indexing_maps = [affine_map<(m, n, k) -> (k, m)>, // transpose
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>]
    ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%cst : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_transposed_A_3D(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = tr %[[arg0]] * %[[arg1]] storeMethod =  set
func.func @matmul_transposed_A_3D(%arg0: tensor<1x2x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x3x3xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x3x3xf32>
  %0 = linalg.batch_matmul
    indexing_maps = [affine_map<(batch, m, n, k) -> (batch, k, m)>, // transpose
                      affine_map<(batch, m, n, k) -> (batch, k, n)>,
                      affine_map<(batch, m, n, k) -> (batch, m, n)>]
    ins(%arg0, %arg1 : tensor<1x2x3xf32>, tensor<1x2x3xf32>) outs(%cst : tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  return %0 : tensor<1x3x3xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_transposed_B_2D(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = %[[arg0]] * tr %[[arg1]] storeMethod =  set
func.func @matmul_transposed_B_2D(%arg0: tensor<1x3x2xf32>, %arg1: tensor<1x3x2xf32>) -> tensor<1x3x3xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x3x3xf32>
  %0 = linalg.batch_matmul
    indexing_maps = [affine_map<(batch, m, n, k) -> (batch, m, k)>,
                      affine_map<(batch, m, n, k) -> (batch, n, k)>, // transpose
                      affine_map<(batch, m, n, k) -> (batch, m, n)>]
    ins(%arg0, %arg1 : tensor<1x3x2xf32>, tensor<1x3x2xf32>) outs(%cst : tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  return %0 : tensor<1x3x3xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_transposed_AB_3D(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = tr %[[arg0]] * tr %[[arg1]] storeMethod =  set
func.func @matmul_transposed_AB_3D(%arg0: tensor<1x3x2xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x2xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x2x2xf32>
  %0 = linalg.batch_matmul
    indexing_maps = [affine_map<(batch, m, n, k) -> (batch, k, m)>, //transpose
                      affine_map<(batch, m, n, k) -> (batch, n, k)>, // transpose
                      affine_map<(batch, m, n, k) -> (batch, m, n)>]
    ins(%arg0, %arg1 : tensor<1x3x2xf32>, tensor<1x2x3xf32>) outs(%cst : tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  return %0 : tensor<1x2x2xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_transposed_AB_2D(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[one:.*]] = rock.gemm %[[zero]] = tr %[[arg0]] * tr %[[arg1]] storeMethod =  set
func.func @matmul_transposed_AB_2D(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x2xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
  %0 = linalg.matmul
    indexing_maps = [affine_map<(m, n, k) -> (k, m)>, //transpose
                      affine_map<(m, n, k) -> (n, k)>, // transpose
                      affine_map<(m, n, k) -> (m, n)>]
    ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%cst : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_perfconf(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[cst:.*]] = arith.constant
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[zero:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[one:.*]] = rock.gemm %[[zero]] = %[[expanded]] * %[[expanded_0]] storeMethod =  set {perf_config = "v2:16,16,8,16,16,4,1,1,1"}
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG: return %[[collapsed]]
func.func @matmul_perfconf(%arg0: tensor<10485760xf16>, %arg1: tensor<52428800xf16>) -> tensor<10485760xf16> attributes {arch = "gfx950", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<512x64x320xf16>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [512, 64, 320] : tensor<10485760xf16> into tensor<512x64x320xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [512, 320, 320] : tensor<52428800xf16> into tensor<512x320x320xf16>
  %0 = linalg.batch_matmul {perf_config = "v2:16,16,8,16,16,4,1,1,1"} ins(%expanded, %expanded_0 : tensor<512x64x320xf16>, tensor<512x320x320xf16>) outs(%cst : tensor<512x64x320xf16>) -> tensor<512x64x320xf16>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<512x64x320xf16> into tensor<10485760xf16>
  return %collapsed : tensor<10485760xf16>
}

// -----

func.func @batch_matmul_broadcast(%arg0: tensor<5xf32>, %arg1: tensor<2x5x7xf32>) -> tensor<2x3x7xf32> attributes {arch = "gfx950", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x3x7xf32>
  // expected-error @+2 {{cannot determine if input matrix is transposed}} 
  // expected-error @+1 {{failed to legalize operation 'linalg.batch_matmul' that was explicitly marked illegal}}
  %0 = linalg.batch_matmul
    indexing_maps = [affine_map<(batch, m, n, k) -> (k)>,           // broadcast
                     affine_map<(batch, m, n, k) -> (batch, k, n)>,
                     affine_map<(batch, m, n, k) -> (batch, m, n)>]
    ins(%arg0, %arg1 : tensor<5xf32>, tensor<2x5x7xf32>)
    outs(%cst: tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
  return %0 : tensor<2x3x7xf32>
}
