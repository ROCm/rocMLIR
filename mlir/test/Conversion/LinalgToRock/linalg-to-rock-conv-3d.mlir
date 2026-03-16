// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -verify-diagnostics --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func.func @conv_3d_basic(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [1 : index, 1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1", "2"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i", "2i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o", "2o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: strides = [1 : index, 1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
module {
  func.func @conv_3d_basic(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<3072xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 10, 10, 10] : tensor<1x3x10x10x10xf32> into tensor<1x1x3x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x8x8x8xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10x10xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst : tensor<1x1x6x8x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x1x6x8x8x8xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4], [5]] : tensor<1x1x6x8x8x8xf32> into tensor<1x6x8x8x8xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3, 4]] : tensor<1x6x8x8x8xf32> into tensor<3072xf32>
    return %collapsed_3 : tensor<3072xf32>
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7 * 2, d4 + d8 * 2, d5 + d9 * 2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func.func @conv_3d_dilation(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [2 : index, 2 : index, 2 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1", "2"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i", "2i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o", "2o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: strides = [1 : index, 1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
module {
  func.func @conv_3d_dilation(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<1296xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 10, 10, 10] : tensor<1x3x10x10x10xf32> into tensor<1x1x3x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x6x6x6xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10x10xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst : tensor<1x1x6x6x6x6xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [2, 2, 2], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x1x6x6x6x6xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4], [5]] : tensor<1x1x6x6x6x6xf32> into tensor<1x6x6x6x6xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3, 4]] : tensor<1x6x6x6x6xf32> into tensor<1296xf32>
    return %collapsed_3 : tensor<1296xf32>
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func.func @conv_3d_padding(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[expanded_3:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_1]], %[[expanded_3]], %[[alloc]])
// CHECK-SAME: dilations = [1 : index, 1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1", "2"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i", "2i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o", "2o"]
// CHECK-SAME: padding = [1 : index, 1 : index, 1 : index, 1 : index, 1 : index, 1 : index]
// CHECK-SAME: strides = [1 : index, 1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_4:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_4]]
module {
  func.func @conv_3d_padding(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<6000xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %expanded_0 low[0, 0, 1, 1, 1] high[0, 0, 1, 1, 1] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f32
    } : tensor<1x3x10x10x10xf32> to tensor<1x3x12x12x12xf32>
    %expanded_1 = tensor.expand_shape %padded [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 12, 12, 12] : tensor<1x3x12x12x12xf32> into tensor<1x1x3x12x12x12xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1x6x10x10x10xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x12x12x12xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst_3 : tensor<1x1x6x10x10x10xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [1, 1, 1, 1, 1, 1], stride = [1, 1, 1]} {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %1 = arith.mulf %in, %in_5 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x1x6x10x10x10xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4], [5]] : tensor<1x1x6x10x10x10xf32> into tensor<1x6x10x10x10xf32>
    %collapsed_4 = tensor.collapse_shape %collapsed [[0, 1, 2, 3, 4]] : tensor<1x6x10x10x10xf32> into tensor<6000xf32>
    return %collapsed_4 : tensor<6000xf32>
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 * 2 + d7, d4 * 2 + d8, d5 * 2 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func.func @conv_3d_stride(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [1 : index, 1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1", "2"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i", "2i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o", "2o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: strides = [2 : index, 2 : index, 2 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
module {
  func.func @conv_3d_stride(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<384xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 10, 10, 10] : tensor<1x3x10x10x10xf32> into tensor<1x1x3x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x4x4x4xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10x10xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst : tensor<1x1x6x4x4x4xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [2, 2, 2]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x1x6x4x4x4xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4], [5]] : tensor<1x1x6x4x4x4xf32> into tensor<1x6x4x4x4xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3, 4]] : tensor<1x6x4x4x4xf32> into tensor<384xf32>
    return %collapsed_3 : tensor<384xf32>
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func.func @conv_3d_groups(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [1 : index, 1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1", "2"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i", "2i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o", "2o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: strides = [1 : index, 1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
module {
  func.func @conv_3d_groups(%arg0: tensor<6000xf32>, %arg1: tensor<486xf32>) -> tensor<4608xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [9, 2, 3, 3, 3] : tensor<486xf32> into tensor<9x2x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 6, 10, 10, 10] : tensor<6000xf32> into tensor<1x6x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 3, 2, 10, 10, 10] : tensor<1x6x10x10x10xf32> into tensor<1x3x2x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [3, 3, 2, 3, 3, 3] : tensor<9x2x3x3x3xf32> into tensor<3x3x2x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x3x3x8x8x8xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x3x2x10x10x10xf32>, tensor<3x3x2x3x3x3xf32>) outs(%cst : tensor<1x3x3x8x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [1, 1, 1], group = 3 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x3x3x8x8x8xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4], [5]] : tensor<1x3x3x8x8x8xf32> into tensor<1x9x8x8x8xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3, 4]] : tensor<1x9x8x8x8xf32> into tensor<4608xf32>
    return %collapsed_3 : tensor<4608xf32>
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 * 2 + d7 * 2, d4 * 2 + d8 * 2, d5 * 2 + d9 * 2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func.func @conv_3d_perf_config(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [2 : index, 2 : index, 2 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1", "2"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i", "2i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o", "2o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"
// CHECK-SAME: strides = [2 : index, 2 : index, 2 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
module {
  func.func @conv_3d_perf_config(%arg0: tensor<750xf32>, %arg1: tensor<96xf32>) -> tensor<64xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [4, 3, 2, 2, 2] : tensor<96xf32> into tensor<4x3x2x2x2xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 3, 5, 5, 5] : tensor<750xf32> into tensor<2x3x5x5x5xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [2, 1, 3, 5, 5, 5] : tensor<2x3x5x5x5xf32> into tensor<2x1x3x5x5x5xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 4, 3, 2, 2, 2] : tensor<4x3x2x2x2xf32> into tensor<1x4x3x2x2x2xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<2x1x4x2x2x2xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<2x1x3x5x5x5xf32>, tensor<1x4x3x2x2x2xf32>) outs(%cst : tensor<2x1x4x2x2x2xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [2, 2, 2], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1", stride = [2, 2, 2]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<2x1x4x2x2x2xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4], [5]] : tensor<2x1x4x2x2x2xf32> into tensor<2x4x2x2x2xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3, 4]] : tensor<2x4x2x2x2xf32> into tensor<64xf32>
    return %collapsed_3 : tensor<64xf32>
  }
}

