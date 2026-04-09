// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock --rock-view-to-transform -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matmul_3D(
// CHECK-NEXT: %[[zero:.*]] = rock.transform
// CHECK-NEXT: %[[one:.*]] = rock.transform
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[two:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[three:.*]] = rock.gemm %[[two]] = %[[one]] * %[[zero]] storeMethod =  set
// CHECK-NEXT: %[[four:.*]] = rock.transform %[[three]]
// CHECK-NEXT: return %[[four]]
func.func @matmul_3D(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<9xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 2, 3] : tensor<6xf32> into tensor<1x2x3xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 3, 2] : tensor<6xf32> into tensor<1x3x2xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<1x3x3xf32>
  %0 = linalg.batch_matmul ins(%expanded_0, %expanded : tensor<1x3x2xf32>, tensor<1x2x3xf32>) outs(%cst : tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<1x3x3xf32> into tensor<9xf32>
  return %collapsed : tensor<9xf32>
}

// CHECK-LABEL: func.func @matmul_2D(
// CHECK-NEXT: %[[zero:.*]] = rock.transform
// CHECK-NEXT: %[[one:.*]] = rock.transform
// CHECK-NEXT: %[[cst:.*]] = arith.constant
// CHECK-NEXT: %[[two:.*]] = bufferization.alloc_tensor
// CHECK-NEXT: %[[three:.*]] = rock.gemm %[[two]] = %[[one]] * %[[zero]] storeMethod =  set
// CHECK-NEXT: %[[four:.*]] = rock.transform %[[three]]
// CHECK-NEXT: return %[[four]]
func.func @matmul_2D(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<9xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [3, 2] : tensor<6xf32> into tensor<3x2xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
  %0 = linalg.matmul ins(%expanded_0, %expanded : tensor<3x2xf32>, tensor<2x3xf32>) outs(%cst : tensor<3x3xf32>) -> tensor<3x3xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<3x3xf32> into tensor<9xf32>
  return %collapsed : tensor<9xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> ((d0 * 3 + d1) * 4 + d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
// CHECK: #map2 = affine_map<(d0) -> (d0 floordiv 6, (d0 mod 6) floordiv 3, d0 mod 3)>
// CHECK: #transform_map = #rock.transform_map<#map by [<Unmerge{2, 3, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 3, 4] -> [24]>
// CHECK: #transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim2", "dim0", "dim1"] at [0, 1, 2] -> ["dim2", "dim0", "dim1"] at [2, 0, 1]>] bounds = [4, 2, 3] -> [2, 3, 4]>
// CHECK: #transform_map2 = #rock.transform_map<#map2 by [<Merge{4, 2, 3} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [24] -> [4, 2, 3]>

// CHECK-LABEL: func.func @transpose_3d(
// CHECK-NEXT: %[[zero:.*]] = rock.transform %{{.*}} by #transform_map
// CHECK-NEXT: %[[empty:.*]] = tensor.empty
// CHECK-NEXT: %[[transposed:.*]] = rock.transform %[[zero]] by #transform_map1
// CHECK-NEXT: %[[two:.*]] = rock.transform %[[transposed]] by #transform_map2
// CHECK-NEXT: return %[[two]]
func.func @transpose_3d(%arg0: tensor<24xf32>) -> tensor<24xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [2, 3, 4] : tensor<24xf32> into tensor<2x3x4xf32>
  %0 = tensor.empty() : tensor<4x2x3xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x3x4xf32>) outs(%0 : tensor<4x2x3xf32>) permutation = [2, 0, 1]
  %collapsed = tensor.collapse_shape %transposed [[0, 1, 2]] : tensor<4x2x3xf32> into tensor<24xf32>
  return %collapsed : tensor<24xf32>
}

// -----

// Making sure regular linalg.generic to untouched
// CHECK-LABEL: some_generic
func.func @some_generic(%arg1: tensor<10x10xf32>) -> tensor<10x10xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %init = tensor.empty() : tensor<10x10xf32>
  // CHECK: linalg.generic
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }
  ins(%arg1 : tensor<10x10xf32>)
  outs(%init: tensor<10x10xf32>){
    ^bb0(%first : f32, %second: f32):
      linalg.yield %first : f32
  } -> tensor<10x10xf32>
  func.return %result : tensor<10x10xf32>
}
