// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg %s | FileCheck %s

// Input: NCDHW = 1x3x10x10x10, Filter: FCDHW = 6x3x3x3x3
// stride=[1,1,1], dilation=[1,1,1], padding=[0,0,0,0,0,0], group=1
// CHECK-LABEL: func.func @conv_3d_basic(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gfchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_3d_basic(%in: !migraphx.shaped<1x3x10x10x10xf32, 3000x1000x100x10x1>, %fil: !migraphx.shaped<6x3x3x3x3xf32, 81x27x9x3x1>) -> !migraphx.shaped<1x6x8x8x8xf32, 3072x512x64x8x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1, 1], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1, 1]} :
    <1x3x10x10x10xf32, 3000x1000x100x10x1>, <6x3x3x3x3xf32, 81x27x9x3x1> -> <1x6x8x8x8xf32, 3072x512x64x8x1>
  func.return %out : !migraphx.shaped<1x6x8x8x8xf32, 3072x512x64x8x1>
}

// -----

// Input: NCDHW = 1x3x10x10x10, Filter: FCDHW = 6x3x3x3x3
// stride=[1,1,1], dilation=[2,2,2], padding=[0,0,0,0,0,0], group=1
// CHECK-LABEL: func.func @conv_3d_dilation(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gfchwd>, dilation = [2, 2, 2], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_3d_dilation(%in: !migraphx.shaped<1x3x10x10x10xf32, 3000x1000x100x10x1>, %fil: !migraphx.shaped<6x3x3x3x3xf32, 81x27x9x3x1>) -> !migraphx.shaped<1x6x6x6x6xf32, 1296x216x36x6x1> {
  %out = migraphx.convolution %in, %fil {dilation = [2, 2, 2], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1, 1]} :
    <1x3x10x10x10xf32, 3000x1000x100x10x1>, <6x3x3x3x3xf32, 81x27x9x3x1> -> <1x6x6x6x6xf32, 1296x216x36x6x1>
  func.return %out : !migraphx.shaped<1x6x6x6x6xf32, 1296x216x36x6x1>
}

// -----

// Input: NCDHW = 1x3x10x10x10, Filter: FCDHW = 6x3x3x3x3
// stride=[1,1,1], dilation=[1,1,1], padding=[1,1,1,1,1,1], group=1
// CHECK-LABEL: func.func @conv_3d_padding(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[padded:.*]] = tensor.pad %[[expanded_0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[padded]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant dense
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gfchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [1, 1, 1, 1, 1, 1], stride = [1, 1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_3d_padding(%in: !migraphx.shaped<1x3x10x10x10xf32, 3000x1000x100x10x1>, %fil: !migraphx.shaped<6x3x3x3x3xf32, 81x27x9x3x1>) -> !migraphx.shaped<1x6x10x10x10xf32, 6000x1000x100x10x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1, 1], group = 1 : i64, padding = [1, 1, 1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1, 1]} :
    <1x3x10x10x10xf32, 3000x1000x100x10x1>, <6x3x3x3x3xf32, 81x27x9x3x1> -> <1x6x10x10x10xf32, 6000x1000x100x10x1>
  func.return %out : !migraphx.shaped<1x6x10x10x10xf32, 6000x1000x100x10x1>
}

// -----

// Input: NCDHW = 1x3x10x10x10, Filter: FCDHW = 6x3x3x3x3
// stride=[2,2,2], dilation=[1,1,1], padding=[0,0,0,0,0,0], group=1
// CHECK-LABEL: func.func @conv_3d_stride(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gfchwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [2, 2, 2]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_3d_stride(%in: !migraphx.shaped<1x3x10x10x10xf32, 3000x1000x100x10x1>, %fil: !migraphx.shaped<6x3x3x3x3xf32, 81x27x9x3x1>) -> !migraphx.shaped<1x6x4x4x4xf32, 384x64x16x4x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1, 1], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} :
    <1x3x10x10x10xf32, 3000x1000x100x10x1>, <6x3x3x3x3xf32, 81x27x9x3x1> -> <1x6x4x4x4xf32, 384x64x16x4x1>
  func.return %out : !migraphx.shaped<1x6x4x4x4xf32, 384x64x16x4x1>
}

// -----

// Input: NCDHW = 1x6x10x10x10, Filter: F(C/G)DHW = 9x2x3x3x3 (group=3, C_per_group=2, F_per_group=3)
// stride=[1,1,1], dilation=[1,1,1], padding=[0,0,0,0,0,0], group=3
// CHECK-LABEL: func.func @conv_3d_groups(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gfchwd>, dilation = [1, 1, 1], group = 3 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_3d_groups(%in: !migraphx.shaped<1x6x10x10x10xf32, 6000x1000x100x10x1>, %fil: !migraphx.shaped<9x2x3x3x3xf32, 54x27x9x3x1>) -> !migraphx.shaped<1x9x8x8x8xf32, 4608x512x64x8x1> {
  %out = migraphx.convolution %in, %fil {dilation = [1, 1, 1], group = 3 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1, 1]} :
    <1x6x10x10x10xf32, 6000x1000x100x10x1>, <9x2x3x3x3xf32, 54x27x9x3x1> -> <1x9x8x8x8xf32, 4608x512x64x8x1>
  func.return %out : !migraphx.shaped<1x9x8x8x8xf32, 4608x512x64x8x1>
}

// -----

// Verifying perf_config attribute passthrough on a kernel function
// CHECK-LABEL: func.func @conv_3d_perf_config(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:     %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType conv3d_ngchwd_gfchwd>, dilation = [2, 2, 2], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1", stride = [2, 2, 2]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[collapsed_0:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:     return %[[collapsed_0]]
func.func @conv_3d_perf_config(%in: !migraphx.shaped<2x3x5x5x5xf32, 375x125x25x5x1>, %fil: !migraphx.shaped<4x3x2x2x2xf32, 24x8x4x2x1>) -> !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1> attributes {kernel, arch="gfx950"} {
  %out = migraphx.convolution %in, %fil {perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1", dilation = [2, 2, 2], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} :
    <2x3x5x5x5xf32, 375x125x25x5x1>, <4x3x2x2x2xf32, 24x8x4x2x1> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  func.return %out : !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>
}
