// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg %s | FileCheck %s

// Input: NCL = 1x3x10, Filter: FCL = 6x3x3
// stride=1, dilation=1, padding=0, group=1
// CHECK-LABEL: func.func @conv_1d_basic(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv1d_ngch_gfch>, dilation = [1], group = 1 : i64, pad = [0, 0], stride = [1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_1d_basic(%in: !migraphx.shaped<1x3x10xf32, 30x10x1>, %fil: !migraphx.shaped<6x3x3xf32, 9x3x1>) -> !migraphx.shaped<1x6x8xf32, 48x8x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1], group = 1 : i64, padding = [0, 0], padding_mode = 0 : i64, stride = [1]} :
    <1x3x10xf32, 30x10x1>, <6x3x3xf32, 9x3x1> -> <1x6x8xf32, 48x8x1>
  func.return %out : !migraphx.shaped<1x6x8xf32, 48x8x1>
}

// -----

// Input: NCL = 1x3x20, Filter: FCL = 6x3x3
// stride=1, dilation=3, padding=0, group=1
// CHECK-LABEL: func.func @conv_1d_dilation(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv1d_ngch_gfch>, dilation = [3], group = 1 : i64, pad = [0, 0], stride = [1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_1d_dilation(%in: !migraphx.shaped<1x3x20xf32, 60x20x1>, %fil: !migraphx.shaped<6x3x3xf32, 9x3x1>) -> !migraphx.shaped<1x6x14xf32, 84x14x1> {
  %out = migraphx.convolution %in, %fil {dilation = [3], group = 1 : i64, padding = [0, 0], padding_mode = 0 : i64, stride = [1]} :
    <1x3x20xf32, 60x20x1>, <6x3x3xf32, 9x3x1> -> <1x6x14xf32, 84x14x1>
  func.return %out : !migraphx.shaped<1x6x14xf32, 84x14x1>
}

// -----

// Input: NCL = 1x3x10, Filter: FCL = 6x3x5
// stride=1, dilation=1, padding=[2,2], group=1
// CHECK-LABEL: func.func @conv_1d_padding(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[padded:.*]] = tensor.pad %[[expanded_0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[padded]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant dense
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv1d_ngch_gfch>, dilation = [1], group = 1 : i64, pad = [2, 2], stride = [1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_1d_padding(%in: !migraphx.shaped<1x3x10xf32, 30x10x1>, %fil: !migraphx.shaped<6x3x5xf32, 15x5x1>) -> !migraphx.shaped<1x6x10xf32, 60x10x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1], group = 1 : i64, padding = [2, 2], padding_mode = 0 : i64, stride = [1]} :
    <1x3x10xf32, 30x10x1>, <6x3x5xf32, 15x5x1> -> <1x6x10xf32, 60x10x1>
  func.return %out : !migraphx.shaped<1x6x10xf32, 60x10x1>
}

// -----

// Input: NCL = 1x3x10, Filter: FCL = 6x3x3
// stride=2, dilation=1, padding=0, group=1
// CHECK-LABEL: func.func @conv_1d_stride(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv1d_ngch_gfch>, dilation = [1], group = 1 : i64, pad = [0, 0], stride = [2]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_1d_stride(%in: !migraphx.shaped<1x3x10xf32, 30x10x1>, %fil: !migraphx.shaped<6x3x3xf32, 9x3x1>) -> !migraphx.shaped<1x6x4xf32, 24x4x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1], group = 1 : i64, padding = [0, 0], padding_mode = 0 : i64, stride = [2]} :
    <1x3x10xf32, 30x10x1>, <6x3x3xf32, 9x3x1> -> <1x6x4xf32, 24x4x1>
  func.return %out : !migraphx.shaped<1x6x4xf32, 24x4x1>
}

// -----

// Input: NCL = 1x6x10, Filter: F(C/G)L = 9x2x3 (group=3, C_per_group=2, F_per_group=3)
// stride=1, dilation=1, padding=0, group=3
// CHECK-LABEL: func.func @conv_1d_groups(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv1d_ngch_gfch>, dilation = [1], group = 3 : i64, pad = [0, 0], stride = [1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_1d_groups(%in: !migraphx.shaped<1x6x10xf32, 60x10x1>, %fil: !migraphx.shaped<9x2x3xf32, 6x3x1>) -> !migraphx.shaped<1x9x8xf32, 72x8x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1], group = 3 : i64, padding = [0, 0], padding_mode = 0 : i64, stride = [1]} :
    <1x6x10xf32, 60x10x1>, <9x2x3xf32, 6x3x1> -> <1x9x8xf32, 72x8x1>
  func.return %out : !migraphx.shaped<1x9x8xf32, 72x8x1>
}
