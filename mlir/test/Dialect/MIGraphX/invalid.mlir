// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

func.func @mlir_reshape_inconsistent_dims(%arg0: !migraphx.shaped<4096x4096xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op dimValue: 64 inconsistent with result dimension 4096}}
  %0 = migraphx.reshape %arg0 {dims = [64, 128]} : <4096x4096xf16, 0x1> -> <4096x4096xf16, 16536x2>
  return
}

func.func @mlir_reshape_dynamic_shape(%arg0: !migraphx.shaped<4096x?xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op Dynamic shapes are not supported}}
  %0 = migraphx.reshape %arg0 {dims = [4096, 4096]} : <4096x?xf16, 0x1> -> <4096x?xf16, 16536x2>
  return
}

func.func @mlir_reshape_rank(%arg0: !migraphx.shaped<4096x4096xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op number of dims (3) does not match result rank (2)}}
  %0 = migraphx.reshape %arg0 {dims = [1, 4096, 4096]} : <4096x4096xf16, 0x1> -> <4096x4096xf16, 16536x2>
  return
}

func.func @mlir_num_input_elements(%arg0: !migraphx.shaped<2x4xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op input and output element counts do not match}}
  %0 = migraphx.reshape %arg0 {dims = [3, 5]} : <2x4xf16, 0x1> -> <3x5xf16, 0x1>
  return
}

func.func @mlir_element_type(%arg0: !migraphx.shaped<2x4xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op failed to verify that all of {input, output} have same element type}}
  %0 = migraphx.reshape %arg0 {dims = [4, 2]} : <2x4xf16, 0x1> -> <4x2xf32, 0x1>
  return
}

func.func @mlir_multiple_neg_one(%arg0: !migraphx.shaped<2x4xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op expected at most one target dimension to be -1}}
  %0 = migraphx.reshape %arg0 {dims = [-1, -1]} : <2x4xf16, 0x1> -> <4x2xf16, 0x1>
  return
}

func.func @mlir_neg_one_with_zero(%arg0: !migraphx.shaped<2x4xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op Cannot mix missing dimensions with zero dimension}}
  %0 = migraphx.reshape %arg0 {dims = [0, -1]} : <2x4xf16, 0x1> -> <4x2xf16, 0x1>
  return
}

func.func @func_equal(%arg0: !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1> attributes{kernel, arch = ""} {
  %cst = migraphx.literal (dense<1> : tensor<1x36x384x64xi32>) : <1x36x384x64xi32, 884736x24576x64x1>
  %0 = migraphx.add %arg0, %cst : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi32, 884736x24576x64x1>
  // expected-error@+1 {{'migraphx.equal' op failed to verify that all of {inA, inB, output} have same element type}}
  %1 = migraphx.equal %arg0, %0 : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi16, 884736x24576x64x1>
  return %1 : !migraphx.shaped<1x36x384x64xi16, 884736x24576x64x1>
}

