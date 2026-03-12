// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg %s | FileCheck %s

// Input: NCHW = 1x3x10x10, Filter: FCHW = 6x3x3x3
// stride=[1,1], dilation=[1,1], padding=[0,0,0,0], group=1
// CHECK-LABEL: func.func @conv_2d_basic(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gfchw>, dilation = [1, 1], group = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_2d_basic(%in: !migraphx.shaped<1x3x10x10xf32, 300x100x10x1>, %fil: !migraphx.shaped<6x3x3x3xf32, 27x9x3x1>) -> !migraphx.shaped<1x6x8x8xf32, 384x64x8x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} :
    <1x3x10x10xf32, 300x100x10x1>, <6x3x3x3xf32, 27x9x3x1> -> <1x6x8x8xf32, 384x64x8x1>
  func.return %out : !migraphx.shaped<1x6x8x8xf32, 384x64x8x1>
}

// -----

// Input: NCHW = 1x3x20x20, Filter: FCHW = 6x3x3x3
// stride=[1,1], dilation=[2,3], padding=[0,0,0,0], group=1
// CHECK-LABEL: func.func @conv_2d_dilation(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gfchw>, dilation = [2, 3], group = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_2d_dilation(%in: !migraphx.shaped<1x3x20x20xf32, 1200x400x20x1>, %fil: !migraphx.shaped<6x3x3x3xf32, 27x9x3x1>) -> !migraphx.shaped<1x6x16x14xf32, 1344x224x14x1> {
  %out = migraphx.convolution %in, %fil {dilation = [2, 3], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} :
    <1x3x20x20xf32, 1200x400x20x1>, <6x3x3x3xf32, 27x9x3x1> -> <1x6x16x14xf32, 1344x224x14x1>
  func.return %out : !migraphx.shaped<1x6x16x14xf32, 1344x224x14x1>
}

// -----

// Input: NCHW = 1x3x10x10, Filter: FCHW = 6x3x3x3
// stride=[1,1], dilation=[1,1], padding=[1,1,1,1], group=1
// CHECK-LABEL: func.func @conv_2d_padding(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[padded:.*]] = tensor.pad %[[expanded_0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[padded]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant dense
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gfchw>, dilation = [1, 1], group = 1 : i64, pad = [1, 1, 1, 1], stride = [1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_2d_padding(%in: !migraphx.shaped<1x3x10x10xf32, 300x100x10x1>, %fil: !migraphx.shaped<6x3x3x3xf32, 27x9x3x1>) -> !migraphx.shaped<1x6x10x10xf32, 600x100x10x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} :
    <1x3x10x10xf32, 300x100x10x1>, <6x3x3x3xf32, 27x9x3x1> -> <1x6x10x10xf32, 600x100x10x1>
  func.return %out : !migraphx.shaped<1x6x10x10xf32, 600x100x10x1>
}

// -----

// Input: NCHW = 1x3x10x10, Filter: FCHW = 6x3x3x3
// stride=[2,3], dilation=[1,1], padding=[0,0,0,0], group=1
// CHECK-LABEL: func.func @conv_2d_stride(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gfchw>, dilation = [1, 1], group = 1 : i64, pad = [0, 0, 0, 0], stride = [2, 3]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_2d_stride(%in: !migraphx.shaped<1x3x10x10xf32, 300x100x10x1>, %fil: !migraphx.shaped<6x3x3x3xf32, 27x9x3x1>) -> !migraphx.shaped<1x6x4x3xf32, 72x12x3x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 3]} :
    <1x3x10x10xf32, 300x100x10x1>, <6x3x3x3xf32, 27x9x3x1> -> <1x6x4x3xf32, 72x12x3x1>
  func.return %out : !migraphx.shaped<1x6x4x3xf32, 72x12x3x1>
}

// -----

// Input: NCHW = 1x6x10x10, Filter: F(C/G)HW = 9x2x3x3 (group=3, C_per_group=2, F_per_group=3)
// stride=[1,1], dilation=[1,1], padding=[0,0,0,0], group=3
// CHECK-LABEL: func.func @conv_2d_groups(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv2d_ngchw_gfchw>, dilation = [1, 1], group = 3 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_2d_groups(%in: !migraphx.shaped<1x6x10x10xf32, 600x100x10x1>, %fil: !migraphx.shaped<9x2x3x3xf32, 18x9x3x1>) -> !migraphx.shaped<1x9x8x8xf32, 576x64x8x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1], group = 3 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} :
    <1x6x10x10xf32, 600x100x10x1>, <9x2x3x3xf32, 18x9x3x1> -> <1x9x8x8xf32, 576x64x8x1>
  func.return %out : !migraphx.shaped<1x9x8x8xf32, 576x64x8x1>
}
