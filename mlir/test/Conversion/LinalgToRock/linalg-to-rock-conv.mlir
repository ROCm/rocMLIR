// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -rock-view-to-transform -verify-diagnostics --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 * 2 + d7 * 2, d4 * 2 + d8 * 2, d5 * 2 + d9 * 2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>

// CHECK-LABEL: func.func @conv_3d(
// CHECK: rock.conv({{.*}}) {dilations = [2 : index, 2 : index, 2 : index], filter_layout = ["g", "k", "0", "1", "2", "c"], input_layout = ["ni", "0i", "1i", "2i", "gi", "ci"], output_layout = ["no", "0o", "1o", "2o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index], strides = [2 : index, 2 : index, 2 : index]}
func.func @conv_3d(%arg0: tensor<64xf32>, %arg1: tensor<750xf32>, %arg2: tensor<96xf32>) -> tensor<64xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x1x4x2x2x2xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [2, 4, 2, 2, 2] : tensor<64xf32> into tensor<2x4x2x2x2xf32>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4, 5]] output_shape [2, 1, 3, 5, 5, 5] : tensor<750xf32> into tensor<2x1x3x5x5x5xf32>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4, 5]] output_shape [1, 4, 3, 2, 2, 2] : tensor<96xf32> into tensor<1x4x3x2x2x2xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_0, %expanded_1 : tensor<2x1x3x5x5x5xf32>, tensor<1x4x3x2x2x2xf32>) outs(%cst : tensor<2x1x4x2x2x2xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gfchwd>, dilation = [2, 2, 2], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [2, 2, 2]} {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %3 = arith.mulf %in, %in_3 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<2x1x4x2x2x2xf32>
  %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4], [5]] : tensor<2x1x4x2x2x2xf32> into tensor<2x4x2x2x2xf32>
  %1 = tensor.empty() : tensor<2x4x2x2x2xf32>
  %2 = linalg.add ins(%collapsed, %expanded : tensor<2x4x2x2x2xf32>, tensor<2x4x2x2x2xf32>) outs(%1 : tensor<2x4x2x2x2xf32>) -> tensor<2x4x2x2x2xf32>
  %collapsed_2 = tensor.collapse_shape %2 [[0, 1, 2, 3, 4]] : tensor<2x4x2x2x2xf32> into tensor<64xf32>
  return %collapsed_2 : tensor<64xf32>
}

// -----

#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 * 4 + d6 * 2, d4 * 5 + d7 * 3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: func.func @conv_2d
// CHECK: rock.conv({{.*}}) {dilations = [2 : index, 3 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [2 : index, 2 : index, 2 : index, 2 : index], strides = [4 : index, 5 : index]}
func.func @conv_2d(%arg0: tensor<122016xf32>, %arg1: tensor<320xf32>) -> tensor<8208xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x2x4x27x19xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 4, 123, 124] : tensor<122016xf32> into tensor<2x4x123x124xf32>
  %padded = tensor.pad %expanded low[0, 0, 2, 2] high[0, 0, 2, 2] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst_0 : f32
  } : tensor<2x4x123x124xf32> to tensor<2x4x127x128xf32>
  %expanded_1 = tensor.expand_shape %padded [[0], [1, 2], [3], [4]] output_shape [2, 2, 2, 127, 128] : tensor<2x4x127x128xf32> into tensor<2x2x2x127x128xf32>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [2, 4, 2, 4, 5] : tensor<320xf32> into tensor<2x4x2x4x5xf32>
  %0 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<2x2x2x127x128xf32>, tensor<2x4x2x4x5xf32>) outs(%cst : tensor<2x2x4x27x19xf32>) attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gfchw>, dilation = [2, 3], group = 2 : i64, pad = [2, 2, 2, 2], stride = [4, 5]} {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %1 = arith.mulf %in, %in_3 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x2x4x27x19xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2, 3, 4]] : tensor<2x2x4x27x19xf32> into tensor<8208xf32>
  return %collapsed : tensor<8208xf32>
}

// -----

#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3 + d5)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv_1d
// CHECK: rock.conv({{.*}}) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "ho", "wo", "go", "ko"], padding = [3 : index, 3 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]}
func.func @conv_1d(%arg0: tensor<14336xf32>, %arg1: tensor<672xf32>, %arg2: tensor<1344xf32>) -> tensor<14336xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x1x64x224xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 3, 224] : tensor<672xf32> into tensor<1x3x224xf32>
  %padded = tensor.pad %expanded low[0, 0, 3] high[0, 0, 3] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x3x224xf32> to tensor<1x3x230xf32>
  %expanded_1 = tensor.expand_shape %padded [[0], [1, 2], [3]] output_shape [1, 1, 3, 230] : tensor<1x3x230xf32> into tensor<1x1x3x230xf32>
  %expanded_2 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 64, 3, 7] : tensor<1344xf32> into tensor<1x64x3x7xf32>
  %0 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x230xf32>, tensor<1x64x3x7xf32>) outs(%cst : tensor<1x1x64x224xf32>) attrs =  {conv_op = #rock<LinalgConvType conv1d_ngch_gfch>, dilation = [1], group = 1 : i64, pad = [3, 3], stride = [1]} {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %1 = arith.mulf %in, %in_3 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<1x1x64x224xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2, 3]] : tensor<1x1x64x224xf32> into tensor<14336xf32>
  return %collapsed : tensor<14336xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 * 2 + d7 * 2, d4 * 2 + d8 * 2, d5 * 2 + d9 * 2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>

// CHECK-LABEL: func.func @mlir_perf_config
// CHECK: rock.conv({{.*}}) {dilations = [2 : index, 2 : index, 2 : index], filter_layout = ["g", "k", "0", "1", "2", "c"], input_layout = ["ni", "0i", "1i", "2i", "gi", "ci"], output_layout = ["no", "0o", "1o", "2o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index], perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1", strides = [2 : index, 2 : index, 2 : index]}
func.func @mlir_perf_config(%arg0: tensor<750xf32>, %arg1: tensor<96xf32>) -> tensor<64xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x1x4x2x2x2xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4, 5]] output_shape [2, 1, 3, 5, 5, 5] : tensor<750xf32> into tensor<2x1x3x5x5x5xf32>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4, 5]] output_shape [1, 4, 3, 2, 2, 2] : tensor<96xf32> into tensor<1x4x3x2x2x2xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded, %expanded_0 : tensor<2x1x3x5x5x5xf32>, tensor<1x4x3x2x2x2xf32>) outs(%cst : tensor<2x1x4x2x2x2xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gfchwd>, dilation = [2, 2, 2], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1", stride = [2, 2, 2]} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.mulf %in, %in_1 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x1x4x2x2x2xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2, 3, 4, 5]] : tensor<2x1x4x2x2x2xf32> into tensor<64xf32>
  return %collapsed : tensor<64xf32>
}
