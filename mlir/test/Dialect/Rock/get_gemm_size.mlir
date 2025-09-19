// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: 2D backward data convolution, no dilation, stride 1, no padding
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @bwd_data_2d_basic
// CHECK: %[[OP:.*]] = "rock.conv_bwd_data"(%[[F:.*]], %[[I:.*]], %[[O:.*]])
// CHECK: %[[SIZE:.*]] = "test.get_gemm_size"(%[[OP]]) : (!rock.conv_bwd_data) -> !rock.gemm_size
// CHECK: return %[[SIZE]]
func.func @bwd_data_2d_basic() -> !rock.gemm_size {
  %filter = "test.tensor"() : () -> tensor<16x3x3x3xf32> // g=16, k=3, y=3, x=3
  %input = "test.tensor"() : () -> tensor<16x3x32x32xf32> // g=16, c=3, hi=32, wi=32
  %output = "test.tensor"() : () -> tensor<16x3x30x30xf32> // g=16, n=3, ho=30, wo=30
  %op = "rock.conv_bwd_data"(%filter, %input, %output)
    { filter_layout = ["g","k","y","x"], input_layout = ["g","c","hi","wi"], output_layout = ["g","n","ho","wo"],
      padding = [0,0,0,0], strides = [1,1], dilations = [1,1], kernel_id = 0 : i64 }
    : (tensor<16x3x3x3xf32>, tensor<16x3x32x32xf32>, tensor<16x3x30x30xf32>) -> tensor<16x3x32x32xf32>
  %size = "test.get_gemm_size"(%op) : (!rock.conv_bwd_data) -> !rock.gemm_size
  return %size : !rock.gemm_size
}

//===----------------------------------------------------------------------===//
// Test: 2D backward data convolution, stride 2, dilation 1, padding 1
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @bwd_data_2d_stride2_pad1
// CHECK: %[[SIZE:.*]] = "test.get_gemm_size"(%[[OP:.*]]) : (!rock.conv_bwd_data) -> !rock.gemm_size
// CHECK: return %[[SIZE]]
func.func @bwd_data_2d_stride2_pad1() -> !rock.gemm_size {
  %filter = "test.tensor"() : () -> tensor<1x8x5x5xf32>
  %input = "test.tensor"() : () -> tensor<1x8x32x32xf32>
  %output = "test.tensor"() : () -> tensor<1x8x16x16xf32>
  %op = "rock.conv_bwd_data"(%filter, %input, %output)
    { filter_layout = ["g","k","y","x"], input_layout = ["g","c","hi","wi"], output_layout = ["g","n","ho","wo"],
      padding = [1,1,1,1], strides = [2,2], dilations = [1,1], kernel_id = 0 : i64 }
    : (tensor<1x8x5x5xf32>, tensor<1x8x32x32xf32>, tensor<1x8x16x16xf32>) -> tensor<1x8x32x32xf32>
  %size = "test.get_gemm_size"(%op) : (!rock.conv_bwd_data) -> !rock.gemm_size
  return %size : !rock.gemm_size
}

//===----------------------------------------------------------------------===//
// Test: 2D backward data convolution, stride 1, dilation 2, padding 0
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @bwd_data_2d_dilation2
// CHECK: %[[SIZE:.*]] = "test.get_gemm_size"(%[[OP:.*]]) : (!rock.conv_bwd_data) -> !rock.gemm_size
// CHECK: return %[[SIZE]]
func.func @bwd_data_2d_dilation2() -> !rock.gemm_size {
  %filter = "test.tensor"() : () -> tensor<2x4x3x3xf32>
  %input = "test.tensor"() : () -> tensor<2x4x32x32xf32>
  %output = "test.tensor"() : () -> tensor<2x4x28x28xf32>
  %op = "rock.conv_bwd_data"(%filter, %input, %output)
    { filter_layout = ["g","k","y","x"], input_layout = ["g","c","hi","wi"], output_layout = ["g","n","ho","wo"],
      padding = [0,0,0,0], strides = [1,1], dilations = [2,2], kernel_id = 0 : i64 }
    : (tensor<2x4x3x3xf32>, tensor<2x4x32x32xf32>, tensor<2x4x28x28xf32>) -> tensor<2x4x32x32xf32>
  %size = "test.get_gemm_size"(%op) : (!rock.conv_bwd_data) -> !rock.gemm_size
  return %size : !rock.gemm_size
}

//===----------------------------------------------------------------------===//
// Test: 3D backward data convolution, stride 1, dilation 1, padding 0
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @bwd_data_3d_basic
// CHECK: %[[SIZE:.*]] = "test.get_gemm_size"(%[[OP:.*]]) : (!rock.conv_bwd_data) -> !rock.gemm_size
// CHECK: return %[[SIZE]]
func.func @bwd_data_3d_basic() -> !rock.gemm_size {
  %filter = "test.tensor"() : () -> tensor<1x2x3x3x3xf32>
  %input = "test.tensor"() : () -> tensor<1x2x16x16x16xf32>
  %output = "test.tensor"() : () -> tensor<1x2x14x14x14xf32>
  %op = "rock.conv_bwd_data"(%filter, %input, %output)
    { filter_layout = ["g","k","y","x","z"], input_layout = ["g","c","hi","wi","di"], output_layout = ["g","n","ho","wo","do"],
      padding = [0,0,0,0,0,0], strides = [1,1,1], dilations = [1,1,1], kernel_id = 0 : i64 }
    : (tensor<1x2x3x3x3xf32>, tensor<1x2x16x16x16xf32>, tensor<1x2x14x14x14xf32>) -> tensor<1x2x16x16x16xf32>
  %size = "test.get_gemm_size"(%op) : (!rock.conv_bwd_data) -> !rock.gemm_size
  return %size : !rock.gemm_size
}

//===----------------------------------------------------------------------===//
// Test: 2D backward data convolution, kernel_id nonzero (for iTilda computation)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @bwd_data_2d_kernelid
// CHECK: %[[SIZE:.*]] = "test.get_gemm_size"(%[[OP:.*]]) : (!rock.conv_bwd_data) -> !rock.gemm_size
// CHECK: return %[[SIZE]]
func.func @bwd_data_2d_kernelid() -> !rock.gemm_size {
  %filter = "test.tensor"() : () -> tensor<1x1x3x3xf32>
  %input = "test.tensor"() : () -> tensor<1x1x8x8xf32>
  %output = "test.tensor"() : () -> tensor<1x1x6x6xf32>
  %op = "rock.conv_bwd_data"(%filter, %input, %output)
    { filter_layout = ["g","k","y","x"], input_layout = ["g","c","hi","wi"], output_layout = ["g","n","ho","wo"],
      padding = [0,0,0,0], strides = [1,1], dilations = [1,1], kernel_id = 3 : i64 }
    : (tensor<1x1x3x3xf32>, tensor<1x1x8x8xf32>, tensor<1x1x6x6xf32>) -> tensor<1x1x8x8xf32>
  %size = "test.get_gemm_size"(%op) : (!rock.conv_bwd_data) -> !rock.gemm_size
  return %size : !rock.gemm_size
}