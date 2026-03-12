// RUN: rocmlir-opt --migraphx-to-linalg -verify-diagnostics %s 

func.func @func_where(%arg0: !migraphx.shaped<1x1xi8, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>, %arg2: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.where'}}
  migraphx.where %arg0, %arg1, %arg2: <1x1xi8, 1x1>, <1x1xf32, 1x1>, <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_convert(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.convert'}}
  migraphx.convert %arg0: <1x1xf32, 1x1> to <1x1xf32, 1x1>
  func.return
}

func.func @func_erf(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.erf'}}
  migraphx.erf %arg0: <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_sigmoid(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.sigmoid'}}
  migraphx.sigmoid %arg0: <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

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

func.func @func_batch_norm_inference(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.batch_norm_inference'}}
  migraphx.batch_norm_inference %arg0, %arg1, %arg1, %arg1, %arg1 {bn_mode = 0 : i64, epsilon = 1.0e-5 : f32, momentum = 0.9 : f32}: 
    !migraphx.shaped<1x1xf32, 1x1>,!migraphx.shaped<1x1xf32, 1x1>,!migraphx.shaped<1x1xf32, 1x1>,!migraphx.shaped<1x1xf32, 1x1>,!migraphx.shaped<1x1xf32, 1x1> -> !migraphx.shaped<1x1xf32, 1x1>
  func.return
}

func.func @func_pooling(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.pooling'}}
  migraphx.pooling %arg0 {mode = "max", padding = [0, 0], stride = [1, 1], ceil_mode = 0 : i64, length = [1, 1]}: <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_flatten(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.flatten'}}
  migraphx.flatten %arg0 {axis = 0 : i64}: <1x1xf32, 1x1> -> <1xf32, 1>
  func.return
}

func.func @func_transpose(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.transpose'}}
  migraphx.transpose %arg0 {permutation = [0, 1]}: <1x1xf32, 1x1> -> <1x1xf32, 1x1>
  func.return
}

func.func @func_slice(%arg0: !migraphx.shaped<1x1xf32, 1x1>, %arg1: !migraphx.shaped<1x1xf32, 1x1>) {
  // expected-error @+1{{failed to legalize operation 'migraphx.slice'}}
  migraphx.slice %arg0 {axes = [0], ends = [1], starts = [0]}: <1x1xf32, 1x1>  -> <1x1xf32, 1x1>
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
