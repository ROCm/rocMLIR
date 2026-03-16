// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -verify-diagnostics --split-input-file | FileCheck %s

// Output: NGCHW = 1x1x1x3x3, Filter: GCKHW = 1x1x1x3x3
// stride=1, dilation=1, padding=[1,1,1,1], group=1

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5, d4, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d4, d2 + d6, d3 + d7)>
// CHECK-LABEL: func.func @mlir_bwd_data_conv(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor
// CHECK-DAG: %[[conv:.*]] = rock.conv_bwd_data(%[[expanded_0]], %[[alloc]], %[[expanded]])
// CHECK-SAME: dilations = [1 : index, 1 : index]
// CHECK-SAME: filter_layout = ["g", "k", "c", "0", "1"]
// CHECK-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i"]
// CHECK-SAME: output_layout = ["no", "go", "ko", "0o", "1o"]
// CHECK-SAME: padding = [1 : index, 1 : index, 1 : index, 1 : index]
// CHECK-SAME: strides = [1 : index, 1 : index]
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[alloc]]
// CHECK-DAG: %[[collapsed_1:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG: return %[[collapsed_1]]
func.func @mlir_bwd_data_conv(%arg0: tensor<9xf32>, %arg1: tensor<9xf32>) -> tensor<9xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1x1x5x5xf32>
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3, 4]] output_shape [1, 1, 1, 3, 3] : tensor<9xf32> into tensor<1x1x1x3x3xf32>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 1, 1, 3, 3] : tensor<9xf32> into tensor<1x1x1x3x3xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%expanded, %expanded_0 : tensor<1x1x1x3x3xf32>, tensor<1x1x1x3x3xf32>) outs(%cst : tensor<1x1x1x5x5xf32>) attrs =  {conv_op = #rock<LinalgConvType convbwd2d_ngchw_gckhw>, dilation = [1, 1], group = 1 : i64, pad = [1, 1, 1, 1], stride = [1, 1]} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
        %1 = arith.mulf %in, %in_2 : f32
        %2 = arith.addf %out, %1 : f32
        linalg.yield %2 : f32
    } -> tensor<1x1x1x5x5xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3], [4]] : tensor<1x1x1x5x5xf32> into tensor<1x1x5x5xf32>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0, 1, 1] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x1x5x5xf32> to tensor<1x1x3x3xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice [[0, 1, 2, 3]] : tensor<1x1x3x3xf32> into tensor<9xf32>
    return %collapsed_1 : tensor<9xf32>
}