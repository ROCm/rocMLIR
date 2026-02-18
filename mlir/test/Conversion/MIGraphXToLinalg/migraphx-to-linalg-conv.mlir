// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK: #[[map:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 * 2 + d7 * 2, d4 * 2 + d8 * 2, d5 * 2 + d9 * 2)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func.func @conv_3d(
// CHECK:         linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins
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

// CHECK-LABEL: func.func @conv_2d(
// CHECK: linalg.conv_2d_ngchw_gfchw
func.func @conv_2d(%arg0: !migraphx.shaped<1x128x28x28xf32, 100352x784x28x1>, %arg1: !migraphx.shaped<1x128x56x56xf32, 401408x3136x56x1>, %arg2: !migraphx.shaped<128x128x3x3xf32, 1152x9x3x1>) -> !migraphx.shaped<1x128x28x28xf32, 100352x784x28x1> {
  %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : <1x128x56x56xf32, 401408x3136x56x1>, <128x128x3x3xf32, 1152x9x3x1> -> <1x128x28x28xf32, 100352x784x28x1>
  %2 = migraphx.add %1, %arg0 : <1x128x28x28xf32, 100352x784x28x1>, <1x128x28x28xf32, 100352x784x28x1> -> <1x128x28x28xf32, 100352x784x28x1>
  %3 = migraphx.relu %2 : <1x128x28x28xf32, 100352x784x28x1> -> <1x128x28x28xf32, 100352x784x28x1>
  return %3 : !migraphx.shaped<1x128x28x28xf32, 100352x784x28x1>
}

// -----
// CHECK: #[[map:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3 + d5)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @conv_1d(
// CHECK:         linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
// CHECK-DAG:       ^bb0(%[[in:.*]]: f32, %[[in_5:.*]]: f32, %[[out:.*]]: f32)
// CHECK-DAG:           %[[three:.*]] = arith.mulf %[[in]], %[[in_5]]
// CHECK-DAG:           %[[four:.*]] = arith.addf %[[out]], %[[three]]
// CHECK-DAG:           linalg.yield %[[four]]
func.func @conv_1d(%arg0: !migraphx.shaped<1x64x224xf32, 14336x224x1>, %arg1: !migraphx.shaped<1x3x224xf32, 672x224x1>, %arg2: !migraphx.shaped<64x3x7xf32, 21x7x1>) -> !migraphx.shaped<1x64x224xf32, 14336x224x1> {
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [1], group = 1 : i64, padding = [3, 3], padding_mode = 0 : i64, stride = [1]} : <1x3x224xf32, 672x224x1>, <64x3x7xf32, 21x7x1> -> <1x64x224xf32, 14336x224x1>
  return %0 : !migraphx.shaped<1x64x224xf32, 14336x224x1>
}

