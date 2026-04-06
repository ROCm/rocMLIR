// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -verify-diagnostics --split-input-file

// expected-error @+1 {{func op does not have the kernel attribute for linalg-to-rock lowering}}
func.func @no_kernel_attribute_test() {
  func.return
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @conv_3d_no_padding(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<3072xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 10, 10, 10] : tensor<1x3x10x10x10xf32> into tensor<1x1x3x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x8x8x8xf32>
    // expected-error @+2 {{no padding found}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10x10xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst : tensor<1x1x6x8x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [1, 1, 1], group = 1 : i64, stride = [1, 1, 1]} {
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

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @conv_3d_no_stride(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<3072xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 10, 10, 10] : tensor<1x3x10x10x10xf32> into tensor<1x1x3x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x8x8x8xf32>
    // expected-error @+2 {{invalid dilation or stride}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10x10xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst : tensor<1x1x6x8x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0]} {
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

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @conv_3d_invalid_padding(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<3072xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 10, 10, 10] : tensor<1x3x10x10x10xf32> into tensor<1x1x3x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x8x8x8xf32>
    // expected-error @+2 {{invalid number of padding}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10x10xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst : tensor<1x1x6x8x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1, 1]} {
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

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @conv_3d_invalid_stride(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<3072xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 10, 10, 10] : tensor<1x3x10x10x10xf32> into tensor<1x1x3x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x8x8x8xf32>
    // expected-error @+2 {{invalid dilation or stride}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10x10xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst : tensor<1x1x6x8x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1]} {
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

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

func.func @conv_1d_padding(%arg0: tensor<30xf32>, %arg1: tensor<90xf32>) -> tensor<60xf32> attributes {kernel, arch="##TOKEN_ARCH##"} {
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [6, 3, 5] : tensor<90xf32> into tensor<6x3x5xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 3, 10] : tensor<30xf32> into tensor<1x3x10xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.empty() : tensor<1x3x14xf32>
  %expanded_1 = tensor.expand_shape %padded [[0], [1, 2], [3]] output_shape [1, 1, 3, 14] : tensor<1x3x14xf32> into tensor<1x1x3x14xf32>
  %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3]] output_shape [1, 6, 3, 5] : tensor<6x3x5xf32> into tensor<1x6x3x5xf32>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1x6x10xf32>
  // expected-error @+2 {{unexpected padding code structure}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x14xf32>, tensor<1x6x3x5xf32>) outs(%cst_3 : tensor<1x1x6x10xf32>) attrs =  {conv_op = #rock<LinalgConvType conv1d_ngch_gkch>, dilation = [1], group = 1 : i64, pad = [2, 2], stride = [1]} {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %1 = arith.mulf %in, %in_5 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<1x1x6x10xf32>
  %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3]] : tensor<1x1x6x10xf32> into tensor<1x6x10xf32>
  %collapsed_4 = tensor.collapse_shape %collapsed [[0, 1, 2]] : tensor<1x6x10xf32> into tensor<60xf32>
  return %collapsed_4 : tensor<60xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @conv_3d_no_dilation(%arg0: tensor<3000xf32>, %arg1: tensor<486xf32>) -> tensor<3072xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [6, 3, 3, 3, 3] : tensor<486xf32> into tensor<6x3x3x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 3, 10, 10, 10] : tensor<3000xf32> into tensor<1x3x10x10x10xf32>
    %expanded_1 = tensor.expand_shape %expanded_0 [[0], [1, 2], [3], [4], [5]] output_shape [1, 1, 3, 10, 10, 10] : tensor<1x3x10x10x10xf32> into tensor<1x1x3x10x10x10xf32>
    %expanded_2 = tensor.expand_shape %expanded [[0, 1], [2], [3], [4], [5]] output_shape [1, 6, 3, 3, 3, 3] : tensor<6x3x3x3x3xf32> into tensor<1x6x3x3x3x3xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x6x8x8x8xf32>
    // expected-error @+2 {{invalid dilation or stride}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%expanded_1, %expanded_2 : tensor<1x1x3x10x10x10xf32>, tensor<1x6x3x3x3x3xf32>) outs(%cst : tensor<1x1x6x8x8x8xf32>) attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gkchwd>, group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1]} {
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
