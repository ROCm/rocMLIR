// RUN: rocmlir-opt --migraphx-to-linalg -verify-diagnostics %s 

func.func @func_unpack(%arg0: !migraphx.shaped<1x1xi8, 1x1>, %arg1: !migraphx.shaped<1x1xi8, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.unpack'}}
  %0 = migraphx.unpack %arg0 {axis = 1 : i64} : <1x1xi8, 1x1> -> <1x2xi8, 2x1>
  func.return
}

func.func @func_quantizelinear(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.quantizelinear'}}
  migraphx.quantizelinear %arg0, %arg1: <1x1xf32, 1x1>, <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_dequantizelinear(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.dequantizelinear'}}
  migraphx.dequantizelinear %arg0, %arg1: <1x1xf32, 1x1>, <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_quant_convolution(%arg0: !migraphx.shaped<1x1xi8, 1x1>, %arg1: !migraphx.shaped<1x1xi8, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.quant_convolution'}}
  migraphx.quant_convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0], stride = [1, 1]}: <1x1xi8, 1x1>, <1x1xi8, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_backwards_data_convolution(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.backwards_data_convolution'}}
  migraphx.backwards_data_convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0], stride = [1, 1]}: <1x1xf32, 1x1>, <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_quant_dot(%arg0: !migraphx.shaped<1x1xf8E4M3FN, 1x1>, %arg1: !migraphx.shaped<1x1xf8E4M3FN, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.quant_dot'}}
  migraphx.quant_dot %arg0, %arg1: <1x1xf8E4M3FN, 1x1>, <1x1xf8E4M3FN, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_softmax(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.softmax'}}
  migraphx.softmax %arg0 {axis= 0:i64}: <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_reduce_mean(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.reduce_mean'}}
  migraphx.reduce_mean %arg0 {axes= [0:i64]}: <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_reduce_max(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.reduce_max'}}
  migraphx.reduce_max %arg0 {axes= [0:i64]} : <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}
