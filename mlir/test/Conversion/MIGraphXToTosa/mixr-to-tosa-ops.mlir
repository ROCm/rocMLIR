// RUN: rocmlir-opt -split-input-file --migraphx-transform --canonicalize --migraphx-to-tosa %s -verify-diagnostics -o -| FileCheck %s

module  {
  // CHECK-LABEL: func @dequantize_scale
  // CHECK-NOT: tosa.sub
  // CHECK: tosa.cast
  // CHECK: tosa.mul
  func.func @dequantize_scale(%arg: !migraphx.shaped<1x112x112x64xi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale : <1x112x112x64xi32, 802816x7168x64x1>, !migraphx.shaped<64xf32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_f32_scale
  // CHECK-NOT: tosa.sub
  // CHECK: tosa.mul
  func.func @dequantize_f32_scale(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, !migraphx.shaped<64xf32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_scale_f16
  // CHECK-NOT: tosa.sub
  // CHECK: tosa.cast{{.*}}f16
  // CHECK: tosa.mul
  func.func @dequantize_scale_f16(%arg: !migraphx.shaped<1x112x112x64xi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf16, 1>) -> !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 =  migraphx.dequantizelinear %arg, %scale : <1x112x112x64xi32, 802816x7168x64x1>, <64xf16, 1> -> <1x112x112x64xf16, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_scale_bias
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub
  // CHECK: tosa.mul
  func.func @dequantize_scale_bias(%arg: !migraphx.shaped<1x112x112x64xi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xi32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xi32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xi32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_wide_bias
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub{{.*}}f32
  // CHECK: tosa.mul
  func.func @dequantize_wide_bias(%arg: !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xi32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xi8, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xi32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_wide_input
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub{{.*}}f32
  // CHECK: tosa.mul
  func.func @dequantize_wide_input(%arg: !migraphx.shaped<1x112x112x64xi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xi8, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xi32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xi8, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_wide_bias_fp8
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub{{.*}}f32
  // CHECK: tosa.mul
  func.func @dequantize_wide_bias_fp8(%arg: !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xf32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_wide_bias_fp8_ocp
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub{{.*}}f32
  // CHECK: tosa.mul
  func.func @dequantize_wide_bias_fp8_ocp(%arg: !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xf8E4M3FN, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xf32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}i8
  // CHECK-NOT: tosa.add
  func.func @quantize_scale(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1> -> <1x112x112x64xi8, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_fp8
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f8E4M3FNUZ
  // CHECK-NOT: tosa.add
  func.func @quantize_scale_fp8(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1> -> <1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_fp8_ocp
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f8E4M3FN
  // CHECK-NOT: tosa.add
  func.func @quantize_scale_fp8_ocp(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1> -> <1x112x112x64xf8E4M3FN, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_bias
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}i8{{.*}}i32
  // CHECK: tosa.cast{{.*}}f32{{.*}}i32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK-SAME: max_int = 127
  // CHECK-SAME: min_int = -128
  // CHECK: tosa.cast{{.*}}i8
  func.func @quantize_scale_bias(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xi8, 1>) -> !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xi8, 1> -> <1x112x112x64xi8, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_bias_fp8
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f8E4M3FNUZ{{.*}}f32
  // CHECK: tosa.cast{{.*}}f32{{.*}}f32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK-SAME: max_fp = 2.400000e+02
  // CHECK-SAME: min_fp = -2.400000e+02
  // CHECK: tosa.cast{{.*}}f8E4M3FNUZ
  func.func @quantize_scale_bias_fp8(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xf8E4M3FNUZ, 1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xf8E4M3FNUZ, 1> -> <1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_bias_fp8_ocp
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f8E4M3FN{{.*}}f32
  // CHECK: tosa.cast{{.*}}f32{{.*}}f32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK-SAME: max_fp = 4.480000e+02
  // CHECK-SAME: min_fp = -4.480000e+02
  // CHECK: tosa.cast{{.*}}f8E4M3FN
  func.func @quantize_scale_bias_fp8_ocp(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xf8E4M3FN, 1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xf8E4M3FN, 1> -> <1x112x112x64xf8E4M3FN, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_bias_f16
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}i8{{.*}}i32
  // CHECK: tosa.cast{{.*}}f16{{.*}}i32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK: tosa.cast
  func.func @quantize_scale_bias_f16(%arg: !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf16, 1>, %bias: !migraphx.shaped<64xi8, 1>) -> !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf16, 802816x7168x64x1>, <64xf16, 1>, !migraphx.shaped<64xi8, 1> -> <1x112x112x64xi8, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_i32_bias_f16
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}i32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK: tosa.cast
  func.func @quantize_scale_i32_bias_f16(%arg: !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf16, 1>, %bias: !migraphx.shaped<64xi32, 1>) -> !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf16, 802816x7168x64x1>, <64xf16, 1>, !migraphx.shaped<64xi32, 1> -> <1x112x112x64xi8, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @conv_with_quant
  // CHECK: tosa.conv2d{{.*}} quantization_info
  // CHECK: tosa.cast
  // CHECK: tosa.cast
  // CHECK: tosa.sub
  // CHECK: tosa.mul
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast
  // CHECK: tosa.cast
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK: tosa.cast
  func.func @conv_with_quant(%arg1: !migraphx.shaped<1x3x224x224xi8, 150528x50176x224x1>, %arg2: !migraphx.shaped<64x3x7x7xi8, 147x49x7x1>, %scale: !migraphx.shaped<1x64x1x1xf32, 64x1x1x1>, %bias: !migraphx.shaped<1x64x1x1xi32, 64x1x1x1>, %bias2: !migraphx.shaped<1x64x1x1xi8, 64x1x1x1>) -> !migraphx.shaped<1x64x112x112xi8, 802816x12544x112x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quant_convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : <1x3x224x224xi8, 150528x50176x224x1>, <64x3x7x7xi8, 147x49x7x1> -> <1x64x112x112xi32, 802816x12544x112x1>
    %2 = migraphx.dequantizelinear %1, %scale, %bias : <1x64x112x112xi32, 802816x12544x112x1>, <1x64x1x1xf32, 64x1x1x1>, !migraphx.shaped<1x64x1x1xi32, 64x1x1x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    %3 = migraphx.quantizelinear %2, %scale, %bias2 : <1x64x112x112xf32, 802816x12544x112x1>, <1x64x1x1xf32, 64x1x1x1>, !migraphx.shaped<1x64x1x1xi8, 64x1x1x1> -> <1x64x112x112xi8, 802816x12544x112x1>
    return %3 : !migraphx.shaped<1x64x112x112xi8, 802816x12544x112x1>
  }

  // CHECK-LABEL: func.func @matmul
  // CHECK: tosa.matmul
  // CHECK-SAME: (tensor<2x256x384xf32>, tensor<2x384x768xf32>) -> tensor<2x256x768xf32>
  func.func @matmul(%arg0: !migraphx.shaped<2x256x384xf32, 98304x384x1>, %arg1: !migraphx.shaped<2x384x768xf32, 294912x768x1>) -> !migraphx.shaped<2x256x768xf32, 196608x768x1> {
    %0 = migraphx.dot %arg0, %arg1 : <2x256x384xf32, 98304x384x1>, <2x384x768xf32, 294912x768x1> -> <2x256x768xf32, 196608x768x1>
     return %0 : !migraphx.shaped<2x256x768xf32, 196608x768x1>
  }

  // CHECK-LABEL: func.func @quant_matmul
  // CHECK: tosa.matmul
  func.func @quant_matmul(%arg0: !migraphx.shaped<2x256x384xi8, 98304x384x1>, %arg1: !migraphx.shaped<2x384x768xi8, 294912x768x1>) -> !migraphx.shaped<2x256x768xi32, 196608x768x1> {
    %0 = migraphx.quant_dot %arg0, %arg1 : <2x256x384xi8, 98304x384x1>, <2x384x768xi8, 294912x768x1> -> <2x256x768xi32, 196608x768x1>
     return %0 : !migraphx.shaped<2x256x768xi32, 196608x768x1>
  }

  // CHECK-LABEL: func.func @quant_matmul_fp8
  // CHECK: tosa.matmul
  func.func @quant_matmul_fp8(%arg0: !migraphx.shaped<1x12x1024x64xf8E4M3FNUZ, 786432x64x768x1>, %arg1: !migraphx.shaped<1x12x64x1024xf8E4M3FNUZ, 786432x64x1x768>) -> !migraphx.shaped<1x12x1024x1024xf32, 12582912x1048576x1024x1> {
    %0 = migraphx.quant_dot %arg0, %arg1 : <1x12x1024x64xf8E4M3FNUZ, 786432x64x768x1>, <1x12x64x1024xf8E4M3FNUZ, 786432x64x1x768> -> <1x12x1024x1024xf32, 12582912x1048576x1024x1>
     return %0 : !migraphx.shaped<1x12x1024x1024xf32, 12582912x1048576x1024x1>
  }

  // CHECK-LABEL: func.func @quant_matmul_fp8_ocp
  // CHECK: tosa.matmul
  func.func @quant_matmul_fp8_ocp(%arg0: !migraphx.shaped<1x12x1024x64xf8E4M3FN, 786432x64x768x1>, %arg1: !migraphx.shaped<1x12x64x1024xf8E4M3FN, 786432x64x1x768>) -> !migraphx.shaped<1x12x1024x1024xf32, 12582912x1048576x1024x1> {
    %0 = migraphx.quant_dot %arg0, %arg1 : <1x12x1024x64xf8E4M3FN, 786432x64x768x1>, <1x12x64x1024xf8E4M3FN, 786432x64x1x768> -> <1x12x1024x1024xf32, 12582912x1048576x1024x1>
     return %0 : !migraphx.shaped<1x12x1024x1024xf32, 12582912x1048576x1024x1>
  }

  // CHECK-LABEL: func.func @matmul_larger_batch
  // CHECK: tosa.matmul
  func.func @matmul_larger_batch(%arg0: !migraphx.shaped<2x16x256x384xf32, 1572864x98304x384x1>, %arg1: !migraphx.shaped<2x16x384x768xf32, 4718592x294912x768x1>) -> !migraphx.shaped<2x16x256x768xf32, 3145728x196608x768x1> {
    %0 = migraphx.dot %arg0, %arg1 : <2x16x256x384xf32, 1572864x98304x384x1>, <2x16x384x768xf32, 4718592x294912x768x1> -> <2x16x256x768xf32, 3145728x196608x768x1>
     return %0 : !migraphx.shaped<2x16x256x768xf32, 3145728x196608x768x1>
  }

  // CHECK-LABEL: func.func @matmul_rank2
  // CHECK: tosa.matmul
  func.func @matmul_rank2(%arg0: !migraphx.shaped<32x72xf32, 72x1>, %arg1: !migraphx.shaped<72x64xf32, 64x1>) -> !migraphx.shaped<32x64xf32, 64x1> {
    %0 = migraphx.dot %arg0, %arg1 : <32x72xf32, 72x1>, <72x64xf32, 64x1> -> <32x64xf32, 64x1>
     return %0 : !migraphx.shaped<32x64xf32, 64x1>
  }

  // CHECK-LABEL: func.func @matmul_broadcast
  func.func @matmul_broadcast(%arg0: !migraphx.shaped<64x64x2304xf16, 147456x2304x1>, %arg1: !migraphx.shaped<64x64x768xf16, 49152x768x1>, %arg2: !migraphx.shaped<1x768x2304xf16, 1769472x2304x1>) -> !migraphx.shaped<64x64x2304xf16, 147456x2304x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 1, 768, 2304>}
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 64, 64, 768>}
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 64, 64, 2304>}
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [64, 768, 2304]} : <1x768x2304xf16, 1769472x2304x1> -> <64x768x2304xf16, 0x2304x1>
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64x768x2304xf16>}> : () -> tensor<64x768x2304xf16>
    // CHECK-DAG: %[[ADD:.*]] = tosa.add %[[CST0]], %[[ARG2]]
    %1 = migraphx.dot %arg1, %0 : <64x64x768xf16, 49152x768x1>, <64x768x2304xf16, 0x2304x1> -> <64x64x2304xf16, 147456x2304x1>
    // CHECK-DAG: %[[MATMUL:.*]] = tosa.matmul %[[ARG1]], %[[ADD]]
    // CHECK-DAG: %[[BIASED:.*]] = tosa.add %[[MATMUL]], %[[ARG0]]
    // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[BIASED]] {new_shape = array<i64: 9437184>}
    // CHECK: return %[[RET]]
    %2 = migraphx.add %1, %arg0 : <64x64x2304xf16, 147456x2304x1>, <64x64x2304xf16, 147456x2304x1> -> <64x64x2304xf16, 147456x2304x1>
    return %2 : !migraphx.shaped<64x64x2304xf16, 147456x2304x1>
  }

  // CHECK-LABEL: func.func @matmul_broadcast_R5
  func.func @matmul_broadcast_R5(%arg0: !migraphx.shaped<2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>, %arg1: !migraphx.shaped<2x4x8x64x768xf16, 1572864x393216x49152x768x1>, %arg2: !migraphx.shaped<1x1x1x768x2304xf16, 1769472x1769472x1769472x2304x1>) -> !migraphx.shaped<2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 1, 1, 1, 768, 2304>}
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 2, 4, 8, 64, 768>}
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 2, 4, 8, 64, 2304>}
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [2, 4, 8, 768, 2304]} : <1x1x1x768x2304xf16, 1769472x1769472x1769472x2304x1> -> <2x4x8x768x2304xf16, 0x0x0x2304x1>
    // CHECK-DAG: %[[RESHAPE0:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 64, 64, 768>}
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x4x8x768x2304xf16>}> : () -> tensor<2x4x8x768x2304xf16>
    // CHECK-DAG: %[[ADD:.*]] = tosa.add %[[CST0]], %[[ARG2]]
    // CHECK-DAG: %[[RESHAPE1:.*]] = tosa.reshape %[[ADD]] {new_shape = array<i64: 64, 768, 2304>}
    %1 = migraphx.dot %arg1, %0 : <2x4x8x64x768xf16, 1572864x393216x49152x768x1>, <2x4x8x768x2304xf16, 0x0x0x2304x1> -> <2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>
    // CHECK-DAG: %[[MATMUL:.*]] = tosa.matmul %[[RESHAPE0]], %[[RESHAPE1]]
    // CHECK: %[[RESHAPE2:.*]] = tosa.reshape %[[MATMUL]] {new_shape = array<i64: 2, 4, 8, 64, 2304>}
    %2 = migraphx.add %1, %arg0 : <2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>, <2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1> -> <2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>
    return %2 : !migraphx.shaped<2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>
  }


  // broadcast ops will be lowered as implicit broadcast in tosa, passes if they're converted and legalize tosa.
  // CHECK-LABEL: func @func_mbcast
  func.func @func_mbcast(%arg0: !migraphx.shaped<1x64x1x1xf32, 64x1x1x1>, %arg1: !migraphx.shaped<1x3x224x224xf32, 150528x50176x224x1>, %arg2: !migraphx.shaped<64x3x7x7xf32, 147x49x7x1>) -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1> attributes {kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg0 {out_lens = [1, 64, 112, 112]} : <1x64x1x1xf32, 64x1x1x1> -> <1x64x112x112xf32, 0x1x0x0>
    %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : <1x3x224x224xf32, 150528x50176x224x1>, <64x3x7x7xf32, 147x49x7x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    %2 = migraphx.add %1, %0 : <1x64x112x112xf32, 802816x12544x112x1>, <1x64x112x112xf32, 0x1x0x0> -> <1x64x112x112xf32, 802816x12544x112x1>
    %3 = migraphx.relu %2 : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    return %3 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
  }

  // CHECK-LABEL: func.func @mbcast_non_first_dim
  // COM: test for a bug in how mbcast was handled in this case.
  // CHECK: new_shape = array<i64: 1, 1, 5, 1>
  func.func @mbcast_non_first_dim(%arg0: !migraphx.shaped<2x3x3x5xf32, 45x15x5x1>, %arg1: !migraphx.shaped<5xf32, 1>) -> !migraphx.shaped<2x3x3x1xf32, 9x3x1x1> attributes {arch = "gfx1100", kernel = "mixr", num_cu = 48 : i64} {
    %0 = migraphx.reshape %arg1 {dims = [5, 1]} : <5xf32, 1> -> <5x1xf32, 1x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 3, 5, 1]} : <5x1xf32, 1x1> -> <2x3x5x1xf32, 0x0x1x1>
    %2 = migraphx.dot %arg0, %1 : <2x3x3x5xf32, 45x15x5x1>, <2x3x5x1xf32, 0x0x1x1> -> <2x3x3x1xf32, 9x3x1x1>
    return %2 : !migraphx.shaped<2x3x3x1xf32, 9x3x1x1>
  }

  // CHECK-LABEL: func.func @clip_i32
  func.func @clip_i32(%arg0: !migraphx.shaped<64x64xi32, 64x1>, %arg1: !migraphx.shaped<64x64xi32, 64x1>, %arg2: !migraphx.shaped<64x64xi32, 64x1>) -> !migraphx.shaped<64x64xi32, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2
    // CHECK: %[[MAX:.*]] = tosa.maximum %[[ARG0]], %[[ARG1]]
    // CHECK: %[[MIN:.*]] = tosa.minimum %[[MAX]], %[[ARG2]]
    // CHECK: %[[RET:.*]] = tosa.reshape %[[MIN]]
    // CHECK: return %[[RET]]
    %0 = migraphx.clip %arg0, %arg1, %arg2 : <64x64xi32, 64x1>, <64x64xi32, 64x1>, <64x64xi32, 64x1> -> <64x64xi32, 64x1>
    return %0 : !migraphx.shaped<64x64xi32, 64x1>
  }

  // CHECK-LABEL: func.func @clip_broadcast
  func.func @clip_broadcast(%arg0: !migraphx.shaped<64x64xf16, 64x1>, %arg1: !migraphx.shaped<1x64xf16, 64x1>, %arg2: !migraphx.shaped<1xf16, 0>) -> !migraphx.shaped<64x64xf16, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 64>}
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 64, 64>}
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64x64xf16>}> : () -> tensor<64x64xf16>
    // CHECK-DAG: %[[ADD0:.*]] = tosa.add %[[CST0]], %[[ARG1]]
    // CHECK-DAG: %[[RESHAPE:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 1, 1>}
    // CHECK-DAG: %[[ADD1:.*]] = tosa.add %[[CST0]], %[[RESHAPE]]
    // CHECK: %[[MAX:.*]] = tosa.maximum %[[ARG0]], %[[ADD0]]
    // CHECK: %[[MIN:.*]] = tosa.minimum %[[MAX]], %[[ADD1]]
    // CHECK: %[[RET:.*]] = tosa.reshape %[[MIN]] {new_shape = array<i64: 4096>}
    // CHECK: return %[[RET]]
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [64, 64]} : <1x64xf16, 64x1> -> <64x64xf16, 0x1>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [64, 64]} : <1xf16, 0> -> <64x64xf16, 0x0>
    %2 = migraphx.clip %arg0, %0, %1 : <64x64xf16, 64x1>, <64x64xf16, 0x1>, <64x64xf16, 0x0> -> <64x64xf16, 64x1>
    return %2 : !migraphx.shaped<64x64xf16, 64x1>
  }

  // CHECK-LABEL: func.func @where
  func.func @where_f32(%arg0: !migraphx.shaped<64x64xi8, 64x1>, %arg1: !migraphx.shaped<64x64xf32, 64x1>, %arg2: !migraphx.shaped<64x64xf32, 64x1>) -> !migraphx.shaped<64x64xf32, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK: tosa.cast
    // CHECK: tosa.select
    %0 = migraphx.where %arg0, %arg1, %arg2 : <64x64xi8, 64x1>, <64x64xf32, 64x1>, <64x64xf32, 64x1> -> <64x64xf32, 64x1>
    return %0 : !migraphx.shaped<64x64xf32, 64x1>
  }

  // CHECK-LABEL: func.func @where_broadcast
  func.func @where_broadcast(%arg0: !migraphx.shaped<64x1xi8, 1x1>, %arg1: !migraphx.shaped<64x64xf16, 64x1>, %arg2: !migraphx.shaped<64x64xf16, 64x1>) -> !migraphx.shaped<64x64xf16, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 64, 1>}
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 64, 64>}
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 64, 64>}
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0> : tensor<64x64xi8>}> : () -> tensor<64x64xi8>
    // CHECK-DAG: %[[ADD:.*]] = tosa.add %[[CST0]], %[[ARG0]]
    // CHECK-DAG: %[[CAST:.*]] = tosa.cast %[[ADD]]
    // CHECK-DAG: tosa.select %[[CAST]], %[[ARG1]], %[[ARG2]]
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [64, 64]} : <64x1xi8, 1x1> -> <64x64xi8, 1x0>
    %1 = migraphx.where %0, %arg1, %arg2 : <64x64xi8, 1x0>, <64x64xf16, 64x1>, <64x64xf16, 64x1> -> <64x64xf16, 64x1>
    return %1 : !migraphx.shaped<64x64xf16, 64x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_f32
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<1.120000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xf32>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_f32(%arg0: !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xf32, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x1x112xf32, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xf32, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_f16
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<1.120000e+02> : tensor<1xf16>}> : () -> tensor<1xf16>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xf16>) -> tensor<1xf16>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xf16>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_f16(%arg0: !migraphx.shaped<1x64x112x112xf16, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xf16, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xf16, 802816x12544x112x1> -> <1x64x1x112xf16, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xf16, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i32
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xi32>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_i32(%arg0: !migraphx.shaped<1x64x112x112xi32, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi32, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xi32, 802816x12544x112x1> -> <1x64x1x112xi32, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi32, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i16
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi16>}> : () -> tensor<1xi16>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xi16>) -> tensor<1xi16>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xi16>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_i16(%arg0: !migraphx.shaped<1x64x112x112xi16, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi16, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xi16, 802816x12544x112x1> -> <1x64x1x112xi16, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi16, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i8
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-DAG: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xi8>) -> tensor<1xi8>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xi8>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_i8(%arg0: !migraphx.shaped<1x64x112x112xi8, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi8, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xi8, 802816x12544x112x1> -> <1x64x1x112xi8, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi8, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_f32
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_f32(%arg0: !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xf32, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x1x112xf32, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xf32, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_f16
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_f16(%arg0: !migraphx.shaped<1x64x112x112xf16, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xf16, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xf16, 802816x12544x112x1> -> <1x64x1x112xf16, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xf16, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_i32
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_i32(%arg0: !migraphx.shaped<1x64x112x112xi32, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi32, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xi32, 802816x12544x112x1> -> <1x64x1x112xi32, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi32, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_i16
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_i16(%arg0: !migraphx.shaped<1x64x112x112xi16, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi16, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xi16, 802816x12544x112x1> -> <1x64x1x112xi16, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi16, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_i8
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-DAG: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_i8(%arg0: !migraphx.shaped<1x64x112x112xi8, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi8, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xi8, 802816x12544x112x1> -> <1x64x1x112xi8, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi8, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_dot_mul
  // CHECK: tosa.matmul
  // CHECK: tosa.mul
  func.func @func_dot_mul(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes{kernel, arch = ""} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %2 = migraphx.mul %0, %arg2 {} : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    return %2 : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }

  // CHECK-LABEL: func.func @func_slice1
  // CHECK: tosa.slice
  func.func @func_slice1(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.slice %arg0 {axes = [1], ends = [12], starts = [0]} : <1x36x384x64xf32, 884736x24576x64x1> -> <1x12x384x64xf32, 294912x24576x64x1>
    return %0 : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_slice2
  // CHECK: tosa.slice
  func.func @func_slice2(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x12x100x64xf32, 76800x6400x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.slice %arg0 {axes = [1, 2], ends = [12, 284], starts = [0, 184]} : <1x36x384x64xf32, 884736x24576x64x1> -> <1x12x100x64xf32, 76800x6400x64x1>
    return %0 : !migraphx.shaped<1x12x100x64xf32, 76800x6400x64x1>
  }
}

// -----

// Unary operations

module {
  // CHECK-LABEL: func.func @func_abs
  // CHECK: tosa.abs
  func.func @func_abs(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.abs %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_ceil
  // CHECK: tosa.ceil
  func.func @func_ceil(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.ceil %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_convert
  // CHECK: tosa.cast
  func.func @func_convert(%arg0: !migraphx.shaped<16xf16, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.convert %arg0 : <16xf16, 1> to <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_convert
  // CHECK: tensor.empty
  // CHECK: linalg.generic
  // CHECK: arith.extui
  func.func @func_convert_int4_unsigned(%arg0: !migraphx.shaped<16xi4, 1>) -> !migraphx.shaped<16xi8, 1> {
    %0 = migraphx.convert zero_extend %arg0 : <16xi4, 1> to <16xi8, 1>
     return %0 : !migraphx.shaped<16xi8, 1>
  }

  // CHECK-LABEL: func.func @func_div_f32
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  func.func @func_div_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xf32, 884736x24576x64x1>, <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_div_f16
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  func.func @func_div_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xf16, 884736x24576x64x1>, <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_div_i32
  // CHECK: tosa.int_div
  func.func @func_div_i32(%arg0: !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_erf_f32
  // CHECK: tosa.erf
  func.func @func_erf_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.erf %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_erf_f16
  // CHECK: tosa.erf
  func.func @func_erf_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.erf %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_exp_f32
  // CHECK: tosa.exp
  func.func @func_exp_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.exp %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_exp_f16
  // CHECK: tosa.exp
  func.func @func_exp_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.exp %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_floor
  // CHECK: tosa.floor
  func.func @func_floor(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.floor %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_log_f32
  // CHECK: tosa.log
  func.func @func_log_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.log %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_log_f16
  // CHECK: tosa.log
  func.func @func_log_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.log %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_neg_f32
  // CHECK: tosa.negate
  func.func @func_neg_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.neg %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_neg_f16
  // CHECK: tosa.negate
  func.func @func_neg_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.neg %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_power
  // CHECK: tosa.pow
  func.func @func_power(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.pow %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
    return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_recip
  // CHECK: tosa.recip
  func.func @func_recip(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.recip %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_rsqrt
  // CHECK: tosa.rsqrt
  func.func @func_rsqrt(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.rsqrt %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_sigmoid
  // CHECK: tosa.sigmoid
  func.func @func_sigmoid(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.sigmoid %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }


  // CHECK-LABEL: func.func @func_rsqrt_opt
  // CHECK: tosa.rsqrt
  // CHECK-NOT: tosa.reciprocal
  func.func @func_rsqrt_opt(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.sqrt %arg0 : <16xf32, 1> -> <16xf32, 1>
    %1 = migraphx.recip %0 : <16xf32, 1> -> <16xf32, 1>
     return %1 : !migraphx.shaped<16xf32, 1>
  }
}

// -----

// Less trivial pointwise ops
module {
  // CHECK-LABEL: func.func @func_softmax_1d
  // CHECK-DAG: [[REDUCE_MAX:%[a-z0-9]+]] = tosa.reduce_max [[INPUT:%[a-z0-9]+]]
  // CHECK-DAG: [[SUB:%[a-z0-9]+]] = tosa.sub [[INPUT]], [[REDUCE_MAX]]
  // CHECK-DAG: [[EXP:%[a-z0-9]+]] = tosa.exp [[SUB]]
  // CHECK-DAG: [[REDUCE_SUM:%[a-z0-9]+]] = tosa.reduce_sum [[EXP]]
  // CHECK-DAG: [[RECIPROCAL:%[a-z0-9]+]] = tosa.reciprocal [[REDUCE_SUM]]
  // CHECK-DAG: tosa.mul [[EXP]], [[RECIPROCAL]]
  func.func @func_softmax_1d(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.softmax %arg0 {axis = 0 : i64} : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_softmax_4d
  // CHECK-DAG: [[REDUCE_MAX:%[a-z0-9]+]] = tosa.reduce_max [[INPUT:%[a-z0-9]+]]
  // CHECK-DAG: [[SUB:%[a-z0-9]+]] = tosa.sub [[INPUT]], [[REDUCE_MAX]]
  // CHECK-DAG: [[EXP:%[a-z0-9]+]] = tosa.exp [[SUB]]
  // CHECK-DAG: [[REDUCE_SUM:%[a-z0-9]+]] = tosa.reduce_sum [[EXP]]
  // CHECK-DAG: [[RECIPROCAL:%[a-z0-9]+]] = tosa.reciprocal [[REDUCE_SUM]]
  // CHECK-DAG: tosa.mul [[EXP]], [[RECIPROCAL]]
  func.func @func_softmax_4d(%arg0: !migraphx.shaped<16x16x16x16xf32, 4096x256x16x1>) -> !migraphx.shaped<16x16x16x16xf32, 4096x256x16x1> {
    %0 = migraphx.softmax %arg0 {axis = 1 : i64} : <16x16x16x16xf32, 4096x256x16x1> -> <16x16x16x16xf32, 4096x256x16x1>
     return %0 : !migraphx.shaped<16x16x16x16xf32, 4096x256x16x1>
  }
}
