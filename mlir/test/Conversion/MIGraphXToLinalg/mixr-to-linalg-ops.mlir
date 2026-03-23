// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg %s -verify-diagnostics  | FileCheck %s

// CHECK-LABEL: func_sub
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.sub ins(%[[expanded_0]], %[[expanded]] {{.*}})
func.func @func_sub(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.sub %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_mul
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.mul ins(%[[expanded_0]], %[[expanded]] {{.*}})
func.func @func_mul(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.mul %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_div
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.div ins(%[[expanded_0]], %[[expanded]] {{.*}})
func.func @func_div(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.div %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_power
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.powf ins(%[[expanded_0]], %[[expanded]] {{.*}})
func.func @func_power(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.pow %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_abs
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.abs ins(%[[expanded_0]] {{.*}})
func.func @func_abs(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.abs %arg0 : <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_ceil
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.ceil ins(%[[expanded]] {{.*}})
func.func @func_ceil(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.ceil %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_exp
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.exp ins(%[[expanded_0]] {{.*}})
func.func @func_exp(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.exp %arg0 : <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_floor
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.floor ins(%[[expanded_0]] {{.*}})
func.func @func_floor(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.floor %arg0 : <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_log
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.log ins(%[[expanded_0]] {{.*}})
func.func @func_log(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.log %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_neg
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.negf ins(%[[expanded_0]] {{.*}})
func.func @func_neg(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.neg %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_sqrt
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.sqrt ins(%[[expanded_0]] {{.*}})
func.func @func_sqrt(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.sqrt %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_tanh
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.tanh ins(%[[expanded_0]] {{.*}})
func.func @func_tanh(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.tanh %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_recip
// CHECK-SAME: (%[[arg0:.*]]: tensor{{.*}}
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG: linalg.reciprocal ins(%[[expanded_0]] {{.*}})
func.func @func_recip(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.recip %arg0: <16xf32, 1> -> <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: func_relu(
// CHECK-SAME: %[[arg0:.*]]: tensor
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:  %[[cst:.*]] = arith.constant
// CHECK-DAG:  %[[cst_0:.*]] = tensor.empty
// CHECK-DAG:  %[[zero:.*]] = linalg.max ins(%[[expanded]], %[[cst]] : {{.*}}) outs(%[[cst_0]] : {{.*}})
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[zero]]
// CHECK-DAG:  return %[[collapsed]]
func.func @func_relu(%arg0: !migraphx.shaped<123x234xf32, 234x1>) -> !migraphx.shaped<123x234xf32, 234x1> {
  %arg1 = migraphx.relu %arg0: <123x234xf32, 234x1> -> <123x234xf32, 234x1>
  func.return %arg1: !migraphx.shaped<123x234xf32, 234x1>
}

// testcase from mixr-to-tosa-ops.mlir

// CHECK-LABEL: @clip_i32(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}}, %[[arg2:.*]]: tensor{{.*}})
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg2]]
// CHECK-DAG:  %[[expanded_0:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:  %[[expanded_1:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:  %[[cst:.*]] = tensor.empty
// CHECK-DAG:  %[[cst_2:.*]] = tensor.empty
// CHECK-DAG:  %[[zero:.*]] = linalg.max ins(%[[expanded_1]], %[[expanded_0]] : {{.*}}) outs(%[[cst]] : {{.*}})
// CHECK-DAG:  %[[one:.*]] = linalg.min ins(%[[zero]], %[[expanded]] : {{.*}}) outs(%[[cst_2]] : {{.*}})
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-DAG:  return %[[collapsed]]
func.func @clip_i32(%arg0: !migraphx.shaped<64x64xi32, 64x1>, %arg1: !migraphx.shaped<64x64xi32, 64x1>, %arg2: !migraphx.shaped<64x64xi32, 64x1>) -> !migraphx.shaped<64x64xi32, 64x1> {
  %0 = migraphx.clip %arg0, %arg1, %arg2 : <64x64xi32, 64x1>, <64x64xi32, 64x1>, <64x64xi32, 64x1> -> <64x64xi32, 64x1>
  return %0 : !migraphx.shaped<64x64xi32, 64x1>
}

  // Literal/Broadcasting test

// CHECK-LABEL: @matmul_broadcast_op(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}}, %[[arg2:.*]]: tensor{{.*}})
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg0]] {{.*}} into tensor<64x64x2304xf16>
// CHECK-DAG:  %[[expanded_0:.*]] = tensor.expand_shape %[[arg1]] {{.*}} into tensor<64x64x768xf16>
// CHECK-DAG:  %[[expanded_1:.*]] = tensor.expand_shape %[[arg2]] {{.*}} into tensor<1x768x2304xf16>
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[expanded_1]] {{.*}} into tensor<1769472xf16>
// CHECK-DAG:  %[[expanded_2:.*]] = tensor.expand_shape %[[collapsed]] {{.*}} into tensor<768x2304xf16>
// CHECK-DAG:  %[[broadcasted:.*]] = linalg.broadcast ins(%[[expanded_2]] : tensor<768x2304xf16>) outs({{.*}} : tensor<64x768x2304xf16>) dimensions = [0]
// CHECK-DAG:  %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64x2304xf16>
// CHECK-DAG:  %[[matmul:.*]] = linalg.batch_matmul ins(%[[expanded_0]], %[[broadcasted]] : {{.*}}) outs(%[[cst]] : {{.*}})
// CHECK-DAG:  %[[add:.*]] = linalg.add ins(%[[matmul]], %[[expanded]] : {{.*}}) outs({{.*}})
// CHECK-DAG:  %[[collapsed_3:.*]] = tensor.collapse_shape %[[add]]
// CHECK-DAG:  return %[[collapsed_3]]
func.func @matmul_broadcast_op(%arg0: !migraphx.shaped<64x64x2304xf16, 147456x2304x1>, %arg1: !migraphx.shaped<64x64x768xf16, 49152x768x1>, %arg2: !migraphx.shaped<1x768x2304xf16, 1769472x2304x1>) -> !migraphx.shaped<64x64x2304xf16, 147456x2304x1> {
  %0 = migraphx.broadcast %arg2 {axis = 0, out_lens = [64, 768, 2304]} : <1x768x2304xf16, 1769472x2304x1> -> <64x768x2304xf16, 0x2304x1>
  %1 = migraphx.dot %arg1, %0 : <64x64x768xf16, 49152x768x1>, <64x768x2304xf16, 0x2304x1> -> <64x64x2304xf16, 147456x2304x1>
  %2 = migraphx.add %1, %arg0 : <64x64x2304xf16, 147456x2304x1>, <64x64x2304xf16, 147456x2304x1> -> <64x64x2304xf16, 147456x2304x1>
  return %2 : !migraphx.shaped<64x64x2304xf16, 147456x2304x1>
}

// CHECK-LABEL: @mbcast_add(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg0]] {{.*}} into tensor<1x64x112x112xf32>
// CHECK-DAG:  %[[expanded_0:.*]] = tensor.expand_shape %[[arg1]] {{.*}} into tensor<1x64x1x1xf32>
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[expanded_0]] {{.*}} into tensor<64xf32>
// CHECK-DAG:  %[[broadcasted:.*]] = linalg.broadcast ins(%[[collapsed]] : tensor<64xf32>) outs({{.*}} : tensor<1x64x112x112xf32>) dimensions = [0, 2, 3]
// CHECK-DAG:  %[[add:.*]] = linalg.add ins(%[[expanded]], %[[broadcasted]] : {{.*}}) outs({{.*}})
// CHECK-DAG:  %[[collapsed_2:.*]] = tensor.collapse_shape %[[add]]
// CHECK-DAG:  return %[[collapsed_2]]
func.func @mbcast_add(
    %arg0: !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>,
    %arg1: !migraphx.shaped<1x64x1x1xf32, 64x1x1x1>
) -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1> {
  %0 = migraphx.multibroadcast %arg1 {out_lens = [1, 64, 112, 112]} : <1x64x1x1xf32, 64x1x1x1> -> <1x64x112x112xf32, 0x1x0x0>
  %1 = migraphx.add %arg0, %0 : <1x64x112x112xf32, 802816x12544x112x1>, <1x64x112x112xf32, 0x1x0x0> -> <1x64x112x112xf32, 802816x12544x112x1>
  return %1 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>

}
// CHECK-LABEL: @literal_splat_f32()
// CHECK-DAG:  %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<4x3xf32>
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[cst]]
// CHECK-DAG:  return %[[collapsed]]
func.func @literal_splat_f32() -> !migraphx.shaped<4x3xf32, 3x1> {
  %0 = migraphx.literal (dense<0.0> : tensor<4x3xf32>) : <4x3xf32, 3x1>
  return %0 : !migraphx.shaped<4x3xf32, 3x1>
}

// CHECK-LABEL: @literal(
// CHECK-SAME: %[[arg0:.*]]: tensor{{.*}})
// CHECK-DAG:  %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<16xf32>
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[cst]]
// CHECK-DAG:  return %[[collapsed]]
func.func @literal(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
  %cst = migraphx.literal (dense<1.0> : tensor<16xf32>) : <16xf32, 1>
  return %cst : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: @literal_dense_si32
// CHECK-DAG:   %[[cst:.*]] = arith.constant dense<{{.*}}> : tensor<2x2xi32>
func.func @literal_dense_si32() -> !migraphx.shaped<2x2xsi32, 2x1> {
  %0 = migraphx.literal (dense<[[0, 1], [2, 3]]> : tensor<2x2xsi32>) : <2x2xsi32, 2x1>
  return %0 : !migraphx.shaped<2x2xsi32, 2x1>
}

// CHECK-LABEL: @scalar_multibroadcast_test
// CHECK-DAG: %[[cst_0:.*]] = arith.constant dense<{{.*}}> : tensor<2x2xf32>
// CHECK-DAG: %[[zero:.*]] = tensor.empty
// CHECK-DAG: %[[one:.*]] = linalg.add ins(%[[cst_0]], %[[cst_0]] : {{.*}}) outs(%[[zero]] : {{.*}})
func.func @scalar_multibroadcast_test() -> !migraphx.shaped<2x2xf32, 2x1> {
  %test = migraphx.literal (dense<0.0> : tensor<f32>) : <f32>
  %result = migraphx.multibroadcast %test {out_dyn_dims = [], out_lens = [2, 2]} : <f32> -> <2x2xf32, 0x0>
  %sum = migraphx.add %result, %result : <2x2xf32, 0x0>, <2x2xf32, 0x0> -> <2x2xf32, 2x1>
  return %sum : !migraphx.shaped<2x2xf32, 2x1>
}

// CHECK-LABEL: @scalar_broadcast_test
// CHECK-DAG:   %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:   %[[zero:.*]] = tensor.empty()
// CHECK-DAG:   %[[broadcasted:.*]] = linalg.broadcast ins(%[[cst]] : {{.*}}) outs(%[[zero]] : {{.*}}) dimensions = [0, 1]
func.func @scalar_broadcast_test() -> !migraphx.shaped<2x2xf32, 2x1> {
  %test = migraphx.literal (dense<0.0> : tensor<f32>) : <f32>
  %result = migraphx.broadcast %test {axis = 1 : i64, out_lens = [2, 2]} : <f32> -> <2x2xf32, 0x0>
  %sum = migraphx.add %result, %result : <2x2xf32, 0x0>, <2x2xf32, 0x0> -> <2x2xf32, 2x1>
  return %sum : !migraphx.shaped<2x2xf32, 2x1>
}

// CHECK-LABEL: @reshape_expand(
// CHECK-SAME: %[[arg0:.*]]: tensor<72xi8>
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg0]] {{.*}} output_shape [9, 8] : tensor<72xi8> into tensor<9x8xi8>
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[expanded]] {{.*}} : tensor<9x8xi8> into tensor<72xi8>
// CHECK-DAG:  %[[expanded_0:.*]] = tensor.expand_shape %[[collapsed]] {{.*}} output_shape [9, 2, 4] : tensor<72xi8> into tensor<9x2x4xi8>
// CHECK-DAG:  %[[empty:.*]] = tensor.empty() : tensor<9x2x4xi8>
// CHECK-DAG:  %[[add:.*]] = linalg.add ins(%[[expanded_0]], %[[expanded_0]] : {{.*}}) outs(%[[empty]] : {{.*}})
// CHECK-DAG:  %[[collapsed_1:.*]] = tensor.collapse_shape %[[add]] {{.*}} : tensor<9x2x4xi8> into tensor<72xi8>
// CHECK-DAG:  return %[[collapsed_1]]
func.func @reshape_expand(%arg0: !migraphx.shaped<9x8xi8, 8x1>) -> !migraphx.shaped<9x2x4xi8, 8x4x1> attributes {arch = "gfx950", kernel} {
  %0 = migraphx.reshape %arg0 {dims = [9, 2, 4]} : <9x8xi8, 8x1> -> <9x2x4xi8, 8x4x1>
  %1 = migraphx.add %0, %0 : <9x2x4xi8, 8x4x1>, <9x2x4xi8, 8x4x1> -> <9x2x4xi8, 8x4x1>
  return %1 : !migraphx.shaped<9x2x4xi8, 8x4x1>
}

// CHECK-LABEL: @reshape_collapse(
// CHECK-SAME: %[[arg0:.*]]: tensor<72xf32>
// CHECK-DAG:  %[[expanded:.*]] = tensor.expand_shape %[[arg0]] {{.*}} output_shape [9, 2, 4] : tensor<72xf32> into tensor<9x2x4xf32>
// CHECK-DAG:  %[[collapsed:.*]] = tensor.collapse_shape %[[expanded]] {{.*}} : tensor<9x2x4xf32> into tensor<72xf32>
// CHECK-DAG:  %[[expanded_0:.*]] = tensor.expand_shape %[[collapsed]] {{.*}} output_shape [9, 8] : tensor<72xf32> into tensor<9x8xf32>
// CHECK-DAG:  %[[empty:.*]] = tensor.empty() : tensor<9x8xf32>
// CHECK-DAG:  %[[add:.*]] = linalg.add ins(%[[expanded_0]], %[[expanded_0]] : {{.*}}) outs(%[[empty]] : {{.*}})
// CHECK-DAG:  %[[collapsed_1:.*]] = tensor.collapse_shape %[[add]] {{.*}} : tensor<9x8xf32> into tensor<72xf32>
// CHECK-DAG:  return %[[collapsed_1]]
func.func @reshape_collapse(%arg0: !migraphx.shaped<9x2x4xf32, 8x4x1>) -> !migraphx.shaped<9x8xf32, 8x1> attributes {arch = "gfx950", kernel} {
  %0 = migraphx.reshape %arg0 {dims = [9, 8]} : <9x2x4xf32, 8x4x1> -> <9x8xf32, 8x1>
  %1 = migraphx.add %0, %0 : <9x8xf32, 8x1>, <9x8xf32, 8x1> -> <9x8xf32, 8x1>
  return %1 : !migraphx.shaped<9x8xf32, 8x1>
}

// -----

// CHECK-LABEL: func.func @transpose_3d
// CHECK-SAME: (%[[arg0:.*]]: tensor<24xf32>
// CHECK: %[[expanded:.*]] = tensor.expand_shape %[[arg0]] {{.*}} into tensor<2x3x4xf32>
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<4x2x3xf32>
// CHECK: %[[transposed:.*]] = linalg.transpose ins(%[[expanded]] : tensor<2x3x4xf32>) outs(%[[empty]] : tensor<4x2x3xf32>) permutation = [2, 0, 1]
// CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[transposed]] {{.*}} : tensor<4x2x3xf32> into tensor<24xf32>
// CHECK: return %[[collapsed]] : tensor<24xf32>
func.func @transpose_3d(%arg0: !migraphx.shaped<2x3x4xf32, 12x4x1>) -> !migraphx.shaped<4x2x3xf32, 6x3x1> {
  %0 = migraphx.transpose %arg0 {permutation = [2, 0, 1]} : <2x3x4xf32, 12x4x1> -> <4x2x3xf32, 6x3x1>
  return %0 : !migraphx.shaped<4x2x3xf32, 6x3x1>
}

// -----
  
// CHECK-LABEL: func.func @func_erf_f32
// CHECK: linalg.erf
func.func @func_erf_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
  %0 = migraphx.erf %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
  return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
}

// CHECK-LABEL: func.func @func_erf_f16
// CHECK: linalg.erf
func.func @func_erf_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
  %0 = migraphx.erf %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
  return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
}