// RUN: rocmlir-opt -split-input-file --migraphx-to-tosa %s | FileCheck %s

// CHECK-LABEL: @migraphx_literal_dense_ui8()
// CHECK-SAME: -> tensor<4xi8> {
func.func @migraphx_literal_dense_ui8() -> !migraphx.shaped<4xui8, 1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{values = dense<[23, 28, 19, 20]> : tensor<4xi8>}> : () -> tensor<4xi8>
  // CHECK-NEXT: return %[[const]] : tensor<4xi8>
  %0 = migraphx.literal (dense<[23, 28, 19, 20]> : tensor<4xui8>) : <4xui8, 1>
  return %0 : !migraphx.shaped<4xui8, 1>
}

// CHECK-LABEL: @migraphx_literal_dense_si8()
// CHECK-SAME: -> tensor<4xi8> {
func.func @migraphx_literal_dense_si8() -> !migraphx.shaped<4xsi8, 1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{values = dense<[-23, 28, -19, 20]> : tensor<4xi8>}> : () -> tensor<4xi8>
  // CHECK-NEXT: return %[[const]] : tensor<4xi8>
  %0 = migraphx.literal (dense<[-23, 28, -19, 20]> : tensor<4xsi8>) : <4xsi8, 1>
  return %0 : !migraphx.shaped<4xsi8, 1>
}

// CHECK-LABEL: @migraphx_literal_dense_i8()
// CHECK-SAME: -> tensor<4xi8> {
func.func @migraphx_literal_dense_i8() -> !migraphx.shaped<4xi8, 1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{values = dense<[-23, 28, -19, 20]> : tensor<4xi8>}> : () -> tensor<4xi8>
  // CHECK-NEXT: return %[[const]] : tensor<4xi8>
  %0 = migraphx.literal (dense<[-23, 28, -19, 20]> : tensor<4xi8>) : <4xi8, 1>
  return %0 : !migraphx.shaped<4xi8, 1>
}

// CHECK-LABEL: @migraphx_literal_dense_f16()
// CHECK-SAME: -> tensor<4xf16> {
func.func @migraphx_literal_dense_f16() -> !migraphx.shaped<4xf16, 1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{values = dense<[-2.300000e+01, 2.800000e+01, -1.900000e+01, 2.000000e+01]> : tensor<4xf16>}> : () -> tensor<4xf16>
  // CHECK-NEXT: return %[[const]] : tensor<4xf16>
  %0 = migraphx.literal (dense<[-23.0, 28.0, -19.0, 20.0]> : tensor<4xf16>) : <4xf16, 1>
  return %0 : !migraphx.shaped<4xf16, 1>
}

// CHECK-LABEL: @migraphx_literal_dense_f32()
// CHECK-SAME: -> tensor<4xf32> {
func.func @migraphx_literal_dense_f32() -> !migraphx.shaped<4xf32, 1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{values = dense<[-2.300000e+01, 2.800000e+01, -1.900000e+01, 2.000000e+01]> : tensor<4xf32>}> : () -> tensor<4xf32>
  // CHECK-NEXT: return %[[const]] : tensor<4xf32>
  %0 = migraphx.literal (dense<[-23.0, 28.0, -19.0, 20.0]> : tensor<4xf32>) : <4xf32, 1>
  return %0 : !migraphx.shaped<4xf32, 1>
}

// CHECK-LABEL: @migraphx_literal_zero()
// CHECK-SAME: -> tensor<9408xi8> {
func.func @migraphx_literal_zero() -> !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{values = dense<0> : tensor<64x3x7x7xi8>}> : () -> tensor<64x3x7x7xi8>
  // CHECK: %[[constshape:.+]] = tosa.const_shape  {values = dense<9408> : tensor<1xindex>} : () -> !tosa.shape<1>
  // CHECK-NEXT: %[[reshape:.+]] = tosa.reshape %[[const]], %[[constshape]] : (tensor<64x3x7x7xi8>, !tosa.shape<1>) -> tensor<9408xi8>
  // CHECK-NEXT: return %[[reshape]] : tensor<9408xi8>
  %0 = migraphx.literal (dense<0> : tensor<64x1xsi8>) : <64x3x7x7xsi8, 147x49x7x1>
  return %0 : !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1>
}

// CHECK-LABEL: @migraphx_literal_negative()
// CHECK-SAME: -> tensor<9408xi8> {
func.func @migraphx_literal_negative() -> !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{values = dense<-1> : tensor<64x3x7x7xi8>}> : () -> tensor<64x3x7x7xi8>
  // CHECK: %[[constshape:.+]] = tosa.const_shape  {values = dense<9408> : tensor<1xindex>} : () -> !tosa.shape<1>
  // CHECK-NEXT: %[[reshape:.+]] = tosa.reshape %[[const]], %[[constshape]] : (tensor<64x3x7x7xi8>, !tosa.shape<1>) -> tensor<9408xi8>
  // CHECK-NEXT: return %[[reshape]] : tensor<9408xi8>
  %0 = migraphx.literal (dense<-1> : tensor<64x1xsi8>) : <64x3x7x7xsi8, 147x49x7x1>
  return %0 : !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1>
}

// CHECK-LABEL: @migraphx_convert_int4_signed
// CHECK: tosa.cast
// CHECK-SAME: (tensor<16xi4>) -> tensor<16xi8>
func.func @migraphx_convert_int4_signed(%arg0: !migraphx.shaped<16xsi4, 1>) -> !migraphx.shaped<16xsi8, 1> {
  %0 = migraphx.convert %arg0 : <16xsi4, 1> to <16xsi8, 1>
  return %0 : !migraphx.shaped<16xsi8, 1>
}

// CHECK-LABEL: @migraphx_convert_int4_unsigned
// CHECK: tosa.custom
// CHECK-SAME: {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xi4>) -> tensor<16xi8>
func.func @migraphx_convert_int4_unsigned(%arg0: !migraphx.shaped<16xui4, 1>) -> !migraphx.shaped<16xui8, 1> {
  %0 = migraphx.convert %arg0 : <16xui4, 1> to <16xui8, 1>
  return %0 : !migraphx.shaped<16xui8, 1>
}

// CHECK-LABEL: @migraphx_convert_int4_unsigned_reverse
// CHECK: tosa.custom
// CHECK-SAME: {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xi8>) -> tensor<16xi4>
func.func @migraphx_convert_int4_unsigned_reverse(%arg0: !migraphx.shaped<16xui8, 1>) -> !migraphx.shaped<16xui4, 1> {
  %0 = migraphx.convert %arg0 : <16xui8, 1> to <16xui4, 1>
  return %0 : !migraphx.shaped<16xui4, 1>
}

// CHECK-LABEL: @migraphx_convert_int4_unsigned_to_float
// CHECK: tosa.custom
// CHECK-SAME: {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xi4>) -> tensor<16xf32>
func.func @migraphx_convert_int4_unsigned_to_float(%arg0: !migraphx.shaped<16xui4, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.convert %arg0 : <16xui4, 1> to <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: @migraphx_convert_int4_float_to_unsigned
// CHECK: tosa.custom
// CHECK-SAME: {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xf32>) -> tensor<16xi4>
func.func @migraphx_convert_int4_float_to_unsigned(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xui4, 1> {
  %0 = migraphx.convert %arg0 : <16xf32, 1> to <16xui4, 1>
  return %0 : !migraphx.shaped<16xui4, 1>
}

// CHECK-LABEL: @migraphx_div_si32
// CHECK: tosa.intdiv
// CHECK-SAME: (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32>
func.func @migraphx_div_si32(%arg0: !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1> attributes{kernel, arch = ""} {
  %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xsi32, 884736x24576x64x1>, <1x36x384x64xsi32, 884736x24576x64x1> -> <1x36x384x64xsi32, 884736x24576x64x1>
  return %0 : !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1>
}

// CHECK-LABEL: @migraphx_div_ui32
// CHECK: tosa.custom
// CHECK-SAME: {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_div"} : (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32>
func.func @migraphx_div_ui32(%arg0: !migraphx.shaped<1x36x384x64xui32, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xui32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xui32, 884736x24576x64x1> attributes{kernel, arch = ""} {
  %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xui32, 884736x24576x64x1>, <1x36x384x64xui32, 884736x24576x64x1> -> <1x36x384x64xui32, 884736x24576x64x1>
  return %0 : !migraphx.shaped<1x36x384x64xui32, 884736x24576x64x1>
}

// CHECK-LABEL: func @dequantize_scale_bias_ui32
// CHECK: tosa.custom %{{.*}} {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<1x112x112x64xi32>) -> tensor<1x112x112x64xf32>
// CHECK: tosa.custom %{{.*}} {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xf32>
// CHECK: tosa.sub
// CHECK: tosa.mul
func.func @dequantize_scale_bias_ui32(%arg: !migraphx.shaped<1x112x112x64xui32, 802816x7168x64x1>, %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>, %bias: !migraphx.shaped<1x1x1x64xui32, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xui32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1>, !migraphx.shaped<1x1x1x64xui32, 64x64x64x1> -> <1x112x112x64xf32, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
}

// CHECK-LABEL: func @dequantize_scale_bias_si32
// CHECK: tosa.cast{{.*}}f32
// CHECK: tosa.cast{{.*}}f32
// CHECK: tosa.sub
// CHECK: tosa.mul
func.func @dequantize_scale_bias_si32(%arg: !migraphx.shaped<1x112x112x64xsi32, 802816x7168x64x1>, %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>, %bias: !migraphx.shaped<1x1x1x64xsi32, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xsi32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1>, !migraphx.shaped<1x1x1x64xsi32, 64x64x64x1> -> <1x112x112x64xf32, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
}

// CHECK-LABEL: func @quantize_scale_bias_ui32
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK: tosa.cast{{.*}}: (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xi32>
// CHECK: tosa.add
func.func @quantize_scale_bias_ui32(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>, %bias: !migraphx.shaped<1x1x1x64xui32, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xui32, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1>, !migraphx.shaped<1x1x1x64xui32, 64x64x64x1> -> <1x112x112x64xui32, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xui32, 802816x7168x64x1>
}

// CHECK-LABEL: func @quantize_scale_bias_si32
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK: tosa.cast{{.*}}f32{{.*}}i32
// CHECK: tosa.add
func.func @quantize_scale_bias_si32(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>, %bias: !migraphx.shaped<1x1x1x64xsi32, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xsi32, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1>, !migraphx.shaped<1x1x1x64xsi32, 64x64x64x1> -> <1x112x112x64xsi32, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xsi32, 802816x7168x64x1>
}

// CHECK-LABEL: func @quantize_scale_bias_ui8
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK: tosa.cast{{.*}}: (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xi32>
// CHECK: tosa.custom{{.*}}i8{{.*}}i32
// CHECK: tosa.add
// CHECK: tosa.clamp{{.*}}i32{{.*}}i32
// CHECK: tosa.custom{{.*}}i32{{.*}}i8
func.func @quantize_scale_bias_ui8(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>, %bias: !migraphx.shaped<1x1x1x64xui8, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xui8, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1>, !migraphx.shaped<1x1x1x64xui8, 64x64x64x1> -> <1x112x112x64xui8, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xui8, 802816x7168x64x1>
}

// CHECK-LABEL: func @quantize_scale_bias_si8
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK: tosa.cast{{.*}}f32{{.*}}i32
// CHECK: tosa.cast{{.*}}i8{{.*}}i32
// CHECK: tosa.add
// CHECK: tosa.clamp{{.*}}i32{{.*}}i32
// CHECK: tosa.cast{{.*}}i32{{.*}}i8
func.func @quantize_scale_bias_si8(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>, %bias: !migraphx.shaped<1x1x1x64xsi8, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xsi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1>, !migraphx.shaped<1x1x1x64xsi8, 64x64x64x1> -> <1x112x112x64xsi8, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xsi8, 802816x7168x64x1>
}

// CHECK-LABEL: func @basic_add_ui32
// CHECK: tosa.add{{.*}}(tensor<1x112x112x64xi32>, tensor<1x112x112x64xi32>) -> tensor<1x112x112x64xi32>
func.func @basic_add_ui32(%arg0: !migraphx.shaped<1x112x112x64xui32, 802816x7168x64x1>, %arg1: !migraphx.shaped<1x112x112x64xui32, 802816x7168x64x1>) -> !migraphx.shaped<1x112x112x64xui32, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.add %arg0, %arg1 : <1x112x112x64xui32, 802816x7168x64x1>, <1x112x112x64xui32, 802816x7168x64x1> -> <1x112x112x64xui32, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xui32, 802816x7168x64x1>
}

// CHECK-LABEL: func @basic_add_si32
// CHECK: tosa.add{{.*}}(tensor<1x112x112x64xi32>, tensor<1x112x112x64xi32>) -> tensor<1x112x112x64xi32>
func.func @basic_add_si32(%arg0: !migraphx.shaped<1x112x112x64xsi32, 802816x7168x64x1>, %arg1: !migraphx.shaped<1x112x112x64xsi32, 802816x7168x64x1>) -> !migraphx.shaped<1x112x112x64xsi32, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.add %arg0, %arg1 : <1x112x112x64xsi32, 802816x7168x64x1>, <1x112x112x64xsi32, 802816x7168x64x1> -> <1x112x112x64xsi32, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xsi32, 802816x7168x64x1>
}

// CHECK-LABEL: func @conv_with_quant_si8
// CHECK: tosa.conv2d{{.*}}(tensor<1x224x224x3xi8>, tensor<64x7x7x3xi8>, tensor<64xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x112x112x64xi32>
// CHECK: tosa.cast{{.*}}(tensor<1x64x112x112xi32>) -> tensor<1x64x112x112xf32>
// CHECK: tosa.cast{{.*}}(tensor<1x64x1x1xi32>) -> tensor<1x64x1x1xf32>
// CHECK: tosa.sub{{.*}}(tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CHECK: tosa.mul{{.*}}(tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>, tensor<1xi8>) -> tensor<1x64x112x112xf32>
// CHECK: tosa.reciprocal{{.*}}(tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
// CHECK: tosa.mul{{.*}}(tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>, tensor<1xi8>) -> tensor<1x64x112x112xf32>
// CHECK: tosa.cast{{.*}}(tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xi32>
// CHECK: tosa.cast{{.*}}(tensor<1x64x1x1xi8>) -> tensor<1x64x1x1xi32>
// CHECK: tosa.add{{.*}}(tensor<1x64x112x112xi32>, tensor<1x64x1x1xi32>) -> tensor<1x64x112x112xi32>
// CHECK: tosa.clamp{{.*}}(tensor<1x64x112x112xi32>) -> tensor<1x64x112x112xi32>
// CHECK: tosa.cast{{.*}}(tensor<1x64x112x112xi32>) -> tensor<1x64x112x112xi8>
func.func @conv_with_quant_si8(%arg1: !migraphx.shaped<1x3x224x224xsi8, 150528x50176x224x1>, %arg2: !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1>, %scale: !migraphx.shaped<1x64x1x1xf32, 64x1x1x1>, %bias: !migraphx.shaped<1x64x1x1xsi32, 64x1x1x1>, %bias2: !migraphx.shaped<1x64x1x1xsi8, 64x1x1x1>) -> !migraphx.shaped<1x64x112x112xsi8, 802816x12544x112x1> attributes {kernel = "mixr"} {
  %1 = migraphx.quant_convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : <1x3x224x224xsi8, 150528x50176x224x1>, <64x3x7x7xsi8, 147x49x7x1> -> <1x64x112x112xsi32, 802816x12544x112x1>
  %2 = migraphx.dequantizelinear %1, %scale, %bias : <1x64x112x112xsi32, 802816x12544x112x1>, <1x64x1x1xf32, 64x1x1x1>, !migraphx.shaped<1x64x1x1xsi32, 64x1x1x1> -> <1x64x112x112xf32, 802816x12544x112x1>
  %3 = migraphx.quantizelinear %2, %scale, %bias2 : <1x64x112x112xf32, 802816x12544x112x1>, <1x64x1x1xf32, 64x1x1x1>, !migraphx.shaped<1x64x1x1xsi8, 64x1x1x1> -> <1x64x112x112xsi8, 802816x12544x112x1>
  return %3 : !migraphx.shaped<1x64x112x112xsi8, 802816x12544x112x1>
}
