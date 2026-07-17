// RUN: rocmlir-opt --rocmlir-custom-tosa-decompose --split-input-file -verify-diagnostics %s

// COM: Negative coverage for verifyConvTranspose in
// COM: mlir/lib/Conversion/RocmlirCustomTosaDecompose/RocmlirCustomTosaDecompose.cpp.
// COM: Each conv_bwd_data tosa.custom op trips one emitOpError branch of the
// COM: transpose-conv verification. Both the strided and non-strided converter
// COM: patterns run verifyConvTranspose, so the branch diagnostic is emitted
// COM: twice, followed by the dialect-conversion "failed to legalize" error.

// COM: stride values must be >= 1
func.func @bwd_data_stride_too_small(%input: tensor<1x3x3x1xf32>, %weight: tensor<1x3x3x1xf32>) -> tensor<1x5x5x1xf32> {
  %b = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error @+2 {{expect all stride values to be >= 1}}
  // expected-error @+1 {{failed to legalize operation 'tosa.custom'}}
  %0 = tosa.custom %input, %weight, %b, %b, %b {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 0, 1>} : (tensor<1x3x3x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
  return %0 : tensor<1x5x5x1xf32>
}

// -----

// COM: out_pad_top must be greater than -KH
func.func @bwd_data_out_pad_top(%input: tensor<1x3x3x1xf32>, %weight: tensor<1x3x3x1xf32>) -> tensor<1x5x5x1xf32> {
  %b = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error @+2 {{expected out_pad_top > -KH}}
  // expected-error @+1 {{failed to legalize operation 'tosa.custom'}}
  %0 = tosa.custom %input, %weight, %b, %b, %b {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: -3, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x3x3x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
  return %0 : tensor<1x5x5x1xf32>
}

// -----

// COM: output height must match the transpose-conv formula
func.func @bwd_data_oh_mismatch(%input: tensor<1x3x3x1xf32>, %weight: tensor<1x3x3x1xf32>) -> tensor<1x6x5x1xf32> {
  %b = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error @+2 {{dimension mismatch: expected OH}}
  // expected-error @+1 {{failed to legalize operation 'tosa.custom'}}
  %0 = tosa.custom %input, %weight, %b, %b, %b {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x3x3x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x6x5x1xf32>
  return %0 : tensor<1x6x5x1xf32>
}

// -----

// COM: output width must match the transpose-conv formula
func.func @bwd_data_ow_mismatch(%input: tensor<1x3x3x1xf32>, %weight: tensor<1x3x3x1xf32>) -> tensor<1x5x6x1xf32> {
  %b = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error @+2 {{dimension mismatch: expected OW}}
  // expected-error @+1 {{failed to legalize operation 'tosa.custom'}}
  %0 = tosa.custom %input, %weight, %b, %b, %b {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x3x3x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x5x6x1xf32>
  return %0 : tensor<1x5x6x1xf32>
}

// -----

// COM: bias channels must equal output channels or be 1
func.func @bwd_data_bias_channels(%input: tensor<1x3x3x1xf32>, %weight: tensor<1x3x3x1xf32>) -> tensor<1x5x5x1xf32> {
  %bias = "tosa.const"() <{values = dense<0.000000e+00> : tensor<3xf32>}> : () -> tensor<3xf32>
  %zp = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error @+2 {{bias channels expected to be equal to output channels}}
  // expected-error @+1 {{failed to legalize operation 'tosa.custom'}}
  %0 = tosa.custom %input, %weight, %bias, %zp, %zp {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x3x3x1xf32>, tensor<1x3x3x1xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x5x5x1xf32>
  return %0 : tensor<1x5x5x1xf32>
}
