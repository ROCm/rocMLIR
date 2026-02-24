// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK: #[[map:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 * 2 + d7 * 2, d4 * 2 + d8 * 2, d5 * 2 + d9 * 2)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func.func @conv_3d(
// CHECK:         linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins
// CHECK-SAME:    attrs = {conv_op = "conv3d_ngchwd_gfchwd", dilation = [2, 2, 2], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [2, 2, 2]}
// CHECK-DAG:       ^bb0(%[[in:.*]]: f32, %[[in_5:.*]]: f32, %[[out:.*]]: f32)
// CHECK-DAG:           %[[three:.*]] = arith.mulf %[[in]], %[[in_5]]
// CHECK-DAG:           %[[four:.*]] = arith.addf %[[out]], %[[three]]
// CHECK-DAG:           linalg.yield %[[four]]
func.func @conv_3d(%arg0: !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>, %arg1: !migraphx.shaped<2x3x5x5x5xf32, 375x125x25x5x1>, %arg2: !migraphx.shaped<4x3x2x2x2xf32, 24x8x4x2x1>) -> !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>  {
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [2, 2, 2], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} : <2x3x5x5x5xf32, 375x125x25x5x1>, <4x3x2x2x2xf32, 24x8x4x2x1> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  %1 = migraphx.add %0, %arg0 : <2x4x2x2x2xf32, 32x8x4x2x1>, <2x4x2x2x2xf32, 32x8x4x2x1> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  return %1 : !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 * 4 + d6 * 2, d4 * 5 + d7 * 3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: func.func @conv_2d(
// CHECK:         linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
// CHECK-SAME:    attrs = {conv_op = "conv2d_ngchw_gfchw", dilation = [2, 3], group = 2 : i64, pad = [2, 2, 2, 2], stride = [4, 5]}
// CHECK-DAG:       ^bb0(%[[in:.*]]: f32, %[[in_5:.*]]: f32, %[[out:.*]]: f32)
// CHECK-DAG:           %[[three:.*]] = arith.mulf %[[in]], %[[in_5]]
// CHECK-DAG:           %[[four:.*]] = arith.addf %[[out]], %[[three]]
// CHECK-DAG:           linalg.yield %[[four]]
func.func @conv_2d(%in: !migraphx.shaped<2x4x123x124xf32, 61008x15252x124x1>, %fil: !migraphx.shaped<8x2x4x5xf32, 40x20x5x1>) -> !migraphx.shaped<2x8x27x19xf32, 4104x513x19x1> {
  %out = migraphx.convolution %in, %fil {dilation = [2, 3], group = 2 : i64, padding = [2, 2, 2, 2], padding_mode = 0 : i64, stride = [4, 5]} : 
    <2x4x123x124xf32, 61008x15252x124x1>, <8x2x4x5xf32, 40x20x5x1> -> <2x8x27x19xf32, 4104x513x19x1>
  func.return %out : !migraphx.shaped<2x8x27x19xf32, 4104x513x19x1>
}

// -----
// CHECK: #[[map:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3 + d5)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @conv_1d(
// CHECK:         linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME:    attrs = {conv_op = "conv1d_ngch_gfch", dilation = [1], group = 1 : i64, pad = [3, 3], stride = [1]}
// CHECK-DAG:       ^bb0(%[[in:.*]]: f32, %[[in_5:.*]]: f32, %[[out:.*]]: f32)
// CHECK-DAG:           %[[three:.*]] = arith.mulf %[[in]], %[[in_5]]
// CHECK-DAG:           %[[four:.*]] = arith.addf %[[out]], %[[three]]
// CHECK-DAG:           linalg.yield %[[four]]
func.func @conv_1d(%arg0: !migraphx.shaped<1x64x224xf32, 14336x224x1>, %arg1: !migraphx.shaped<1x3x224xf32, 672x224x1>, %arg2: !migraphx.shaped<64x3x7xf32, 21x7x1>) -> !migraphx.shaped<1x64x224xf32, 14336x224x1> {
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [1], group = 1 : i64, padding = [3, 3], padding_mode = 0 : i64, stride = [1]} : <1x3x224xf32, 672x224x1>, <64x3x7xf32, 21x7x1> -> <1x64x224xf32, 14336x224x1>
  return %0 : !migraphx.shaped<1x64x224xf32, 14336x224x1>
}

// -----

// Currently, we don't support type casting
func.func @conv_1d_different_types(%arg1: !migraphx.shaped<1x3x224xf16, 672x224x1>, %arg2: !migraphx.shaped<64x3x7xf16, 21x7x1>) -> !migraphx.shaped<1x64x224xf32, 14336x224x1> {
  // expected-error @+2 {{type casting between operands and result is unsupported for now}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [1], group = 1 : i64, padding = [3, 3], padding_mode = 0 : i64, stride = [1]} : <1x3x224xf16, 672x224x1>, <64x3x7xf16, 21x7x1> -> <1x64x224xf32, 14336x224x1>
  return %0 : !migraphx.shaped<1x64x224xf32, 14336x224x1>
}

// -----

// Checking for the perf_config, dilation, strides, and pad attributes

// CHECK-LABEL: func.func @mlir_convolution_add(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}}
// CHECK-DAG:   %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:   %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:   %[[expanded_1:.*]] = tensor.expand_shape %[[expanded_0]]
// CHECK-DAG:   %[[expanded_2:.*]] = tensor.expand_shape %[[expanded]]
// CHECK-DAG:   %[[cst:.*]] = arith.constant
// CHECK-DAG:   %[[zero:.*]] = linalg.generic {{.*}} ins(%[[expanded_1]], %[[expanded_2]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:   attrs =  {conv_op = "conv3d_ngchwd_gfchwd", dilation = [2, 2, 2], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1", stride = [2, 2, 2]}
// CHECK-DAG:   %[[collapsed:.*]] = tensor.collapse_shape %[[zero]]
// CHECK-DAG:   %[[collapsed_3:.*]] = tensor.collapse_shape %[[collapsed]]
// CHECK-DAG:   return %[[collapsed_3]]
func.func @mlir_convolution_add(%arg1: !migraphx.shaped<2x3x5x5x5xf32, 375x125x25x5x1>, %arg2: !migraphx.shaped<4x3x2x2x2xf32, 24x8x4x2x1>) -> !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1> attributes {kernel, arch="gfx950"}{
  %0 = migraphx.convolution %arg1, %arg2 {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1", dilation = [2, 2, 2], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} : <2x3x5x5x5xf32, 375x125x25x5x1>, <4x3x2x2x2xf32, 24x8x4x2x1> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  return %0 : !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>
}
