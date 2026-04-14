// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -verify-diagnostics --split-input-file --remove-dead-values | FileCheck %s

// CHECK-LABEL: func.func @conv_2d_basic(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: strides = [1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module {
  func.func @conv_2d_basic(%arg0: tensor<300xf32>, %arg1: tensor<162xf32>) -> tensor<384xf32> attributes {rock.kernel, rock.arch="##TOKEN_ARCH##"}{
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [6, 3, 3, 3] : tensor<162xf32> into tensor<6x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 3, 10, 10] : tensor<300xf32> into tensor<1x3x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4]] output_shape [1, 1, 3, 10, 10] : tensor<1x3x10x10xf32> into tensor<1x1x3x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4]] output_shape [1, 6, 3, 3, 3] : tensor<6x3x3x3xf32> into tensor<1x6x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x8x8xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10xf32>, tensor<1x6x3x3x3xf32>) outs(%cst : tensor<1x1x6x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gkchw>, dilation = [1, 1], group = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x1x6x8x8xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4]] : tensor<1x1x6x8x8xf32> into tensor<1x6x8x8xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3]] : tensor<1x6x8x8xf32> into tensor<384xf32>
    return %collapsed_3 : tensor<384xf32>
  }
}

// -----

// CHECK-LABEL: func.func @conv_2d_dilation(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [2 : index, 3 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: strides = [1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6 * 2, d4 + d7 * 3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module {
  func.func @conv_2d_dilation(%arg0: tensor<1200xf32>, %arg1: tensor<162xf32>) -> tensor<1344xf32> attributes {rock.kernel, rock.arch="##TOKEN_ARCH##"} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [6, 3, 3, 3] : tensor<162xf32> into tensor<6x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 3, 20, 20] : tensor<1200xf32> into tensor<1x3x20x20xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4]] output_shape [1, 1, 3, 20, 20] : tensor<1x3x20x20xf32> into tensor<1x1x3x20x20xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4]] output_shape [1, 6, 3, 3, 3] : tensor<6x3x3x3xf32> into tensor<1x6x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x16x14xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x20x20xf32>, tensor<1x6x3x3x3xf32>) outs(%cst : tensor<1x1x6x16x14xf32>) attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gkchw>, dilation = [2, 3], group = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x1x6x16x14xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4]] : tensor<1x1x6x16x14xf32> into tensor<1x6x16x14xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3]] : tensor<1x6x16x14xf32> into tensor<1344xf32>
    return %collapsed_3 : tensor<1344xf32>
  }
}

// -----

// CHECK-LABEL: func.func @conv_2d_padding(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[expanded_3:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_1]], %[[expanded_3]], %[[alloc]])
// CHECK-SAME: dilations = [1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o"]
// CHECK-SAME: padding = [1 : index, 1 : index, 1 : index, 1 : index]
// CHECK-SAME: strides = [1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_4:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_4]]
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module {
  func.func @conv_2d_padding(%arg0: tensor<300xf32>, %arg1: tensor<162xf32>) -> tensor<600xf32> attributes {rock.kernel, rock.arch="##TOKEN_ARCH##"} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [6, 3, 3, 3] : tensor<162xf32> into tensor<6x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 3, 10, 10] : tensor<300xf32> into tensor<1x3x10x10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %expanded_0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst : f32
    } : tensor<1x3x10x10xf32> to tensor<1x3x12x12xf32>
    %expanded_1 = tensor.expand_shape %padded [[0], [1, 2], [3], [4]] output_shape [1, 1, 3, 12, 12] : tensor<1x3x12x12xf32> into tensor<1x1x3x12x12xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4]] output_shape [1, 6, 3, 3, 3] : tensor<6x3x3x3xf32> into tensor<1x6x3x3x3xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1x6x10x10xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x12x12xf32>, tensor<1x6x3x3x3xf32>) outs(%cst_3 : tensor<1x1x6x10x10xf32>) attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gkchw>, dilation = [1, 1], group = 1 : i64, pad = [1, 1, 1, 1], stride = [1, 1]} {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %1 = arith.mulf %in, %in_5 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x1x6x10x10xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4]] : tensor<1x1x6x10x10xf32> into tensor<1x6x10x10xf32>
    %collapsed_4 = tensor.collapse_shape %collapsed [[0, 1, 2, 3]] : tensor<1x6x10x10xf32> into tensor<600xf32>
    return %collapsed_4 : tensor<600xf32>
  }
}

// -----

// CHECK-LABEL: func.func @conv_2d_stride(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: strides = [2 : index, 3 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 * 2 + d6, d4 * 3 + d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module {
  func.func @conv_2d_stride(%arg0: tensor<300xf32>, %arg1: tensor<162xf32>) -> tensor<72xf32> attributes {rock.kernel, rock.arch="##TOKEN_ARCH##"} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [6, 3, 3, 3] : tensor<162xf32> into tensor<6x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 3, 10, 10] : tensor<300xf32> into tensor<1x3x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4]] output_shape [1, 1, 3, 10, 10] : tensor<1x3x10x10xf32> into tensor<1x1x3x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4]] output_shape [1, 6, 3, 3, 3] : tensor<6x3x3x3xf32> into tensor<1x6x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x4x3xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10xf32>, tensor<1x6x3x3x3xf32>) outs(%cst : tensor<1x1x6x4x3xf32>) attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gkchw>, dilation = [1, 1], group = 1 : i64, pad = [0, 0, 0, 0], stride = [2, 3]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x1x6x4x3xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4]] : tensor<1x1x6x4x3xf32> into tensor<1x6x4x3xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3]] : tensor<1x6x4x3xf32> into tensor<72xf32>
    return %collapsed_3 : tensor<72xf32>
  }
}

// -----

// CHECK-LABEL: func.func @conv_2d_groups(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG: %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv(%[[expanded_2]], %[[expanded_1]], %[[alloc]])
// CHECK-SAME: dilations = [1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o"]
// CHECK-SAME: padding = [0 : index, 0 : index, 0 : index, 0 : index]
// CHECK-SAME: strides = [1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG: %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_3]]
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module {
  func.func @conv_2d_groups(%arg0: tensor<600xf32>, %arg1: tensor<162xf32>) -> tensor<576xf32> attributes {rock.kernel, rock.arch="##TOKEN_ARCH##"} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [9, 2, 3, 3] : tensor<162xf32> into tensor<9x2x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 6, 10, 10] : tensor<600xf32> into tensor<1x6x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4]] output_shape [1, 3, 2, 10, 10] : tensor<1x6x10x10xf32> into tensor<1x3x2x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4]] output_shape [3, 3, 2, 3, 3] : tensor<9x2x3x3xf32> into tensor<3x3x2x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x3x3x8x8xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x3x2x10x10xf32>, tensor<3x3x2x3x3xf32>) outs(%cst : tensor<1x3x3x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gkchw>, dilation = [1, 1], group = 3 : i64, pad = [0, 0, 0, 0], stride = [1, 1]} {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x3x3x8x8xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4]] : tensor<1x3x3x8x8xf32> into tensor<1x9x8x8xf32>
    %collapsed_3 = tensor.collapse_shape %collapsed [[0, 1, 2, 3]] : tensor<1x9x8x8xf32> into tensor<576xf32>
    return %collapsed_3 : tensor<576xf32>
  }
}
