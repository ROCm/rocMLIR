// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s

// 4D spatial convolution is not supported
func.func @conv_4d_unsupported(%in: !migraphx.shaped<1x3x5x5x5x5xf32, 1875x625x125x25x5x1>, %fil: !migraphx.shaped<6x3x3x3x3x3xf32, 243x81x27x9x3x1>) -> !migraphx.shaped<1x6x3x3x3x3xf32, 486x81x27x9x3x1> {
  // expected-error @+2 {{4D conv is not supported for now}}
  // expected-error @+1 {{failed to legalize operation}}
  %out = migraphx.convolution %in, %fil {dilation = [1, 1, 1, 1], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1, 1, 1]} :
    <1x3x5x5x5x5xf32, 1875x625x125x25x5x1>, <6x3x3x3x3x3xf32, 243x81x27x9x3x1> -> <1x6x3x3x3x3xf32, 486x81x27x9x3x1>
  func.return %out : !migraphx.shaped<1x6x3x3x3x3xf32, 486x81x27x9x3x1>
}

// -----

// Type casting between operands and result is not supported
func.func @conv_1d_different_types(%in: !migraphx.shaped<1x3x224xf16, 672x224x1>, %fil: !migraphx.shaped<64x3x7xf16, 21x7x1>) -> !migraphx.shaped<1x64x224xf32, 14336x224x1> {
  // expected-error @+2 {{type casting between operands and result is unsupported for now}}
  // expected-error @+1 {{failed to legalize operation}}
  %out = migraphx.convolution %in, %fil {dilation = [1], group = 1 : i64, padding = [3, 3], padding_mode = 0 : i64, stride = [1]} :
    <1x3x224xf16, 672x224x1>, <64x3x7xf16, 21x7x1> -> <1x64x224xf32, 14336x224x1>
  func.return %out : !migraphx.shaped<1x64x224xf32, 14336x224x1>
}
