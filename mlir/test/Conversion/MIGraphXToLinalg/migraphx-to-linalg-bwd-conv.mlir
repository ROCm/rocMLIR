// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg --canonicalize --cse --remove-dead-values %s | FileCheck %s

// CHECK-LABEL: func.func @mlir_bwd_data_conv(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded]], %[[expanded_0]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType convbwd2d_ngchw_gckhw>, dilation = [1, 1], group = 1 : i64, pad = [1, 1, 1, 1], stride = [2, 3]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     %[[extracted_slice:.*]] = tensor.extract_slice %[[collapsed]]
// CHECK-DAG:     %[[collapsed_1:.*]] = tensor.collapse_shape %[[extracted_slice]]
// CHECK-DAG:     return %[[collapsed_1]]
func.func @mlir_bwd_data_conv(
    %arg0: !migraphx.shaped<1x3x6x7xf32, 126x42x7x1>,
    %arg1: !migraphx.shaped<3x4x3x3xf32, 36x9x3x1>
) -> !migraphx.shaped<1x4x11x19xf32, 836x209x19x1> {
  %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
    dilation = [1, 1],
    group = 1 : i64,
    padding = [1, 1, 1, 1],
    padding_mode = 0 : i64,
    stride = [2, 3]} : <1x3x6x7xf32, 126x42x7x1>, <3x4x3x3xf32, 36x9x3x1> -> <1x4x11x19xf32, 836x209x19x1>
  return %0 : !migraphx.shaped<1x4x11x19xf32, 836x209x19x1>
}

// -----

// Output grad: NCDHW = 1x1x1x3x3, Filter: CKDHW = 1x1x1x3x3
// stride=[1,1,1], dilation=[1,1,1], padding=[0,0,0,0,0,0], group=1
// CHECK-LABEL: func.func @mlir_bwd_data_conv(
// CHECK-SAME:  %[[arg0:.*]]: tensor{{.*}}, %[[arg1:.*]]: tensor{{.*}})
// CHECK-DAG:     %[[cst:.*]] = arith.constant
// CHECK-DAG:     %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-DAG:     %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-DAG:     %[[conv:.*]] = linalg.generic {{.*}} ins(%[[expanded]], %[[expanded_0]] : tensor{{.*}}) outs(%[[cst]] : tensor{{.*}})
// CHECK-SAME:      attrs =  {conv_op = #rock<LinalgConvType convbwd3d_ngchwd_gckhwd>, dilation = [1, 1, 1], group = 1 : i64, pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]}
// CHECK-DAG:     %[[collapsed:.*]] = tensor.collapse_shape %[[conv]]
// CHECK-DAG:     return %[[collapsed]]
func.func @mlir_bwd_data_conv(
    %arg0: !migraphx.shaped<1x1x1x3x3xf32, 9x9x9x3x1>,
    %arg1: !migraphx.shaped<1x1x1x3x3xf32, 9x9x9x3x1>
) -> !migraphx.shaped<1x1x1x5x5xf32, 25x25x25x5x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = migraphx.backwards_data_convolution %arg1, %arg0 {
    dilation = [1, 1, 1],
    group = 1 : i64,
    padding = [0, 0, 0, 0, 0, 0],
    padding_mode = 0 : i64,
    stride = [1, 1, 1]
  } : <1x1x1x3x3xf32, 9x9x9x3x1>, <1x1x1x3x3xf32, 9x9x9x3x1> -> <1x1x1x5x5xf32, 25x25x25x5x1>
  return %0 : !migraphx.shaped<1x1x1x5x5xf32, 25x25x25x5x1>
}
