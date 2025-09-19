// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: ConvBwdWeightOp::getOutArgument returns the first operand (filter)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @get_out_argument_basic
// CHECK: %[[F:.*]] = "test.tensor"() : () -> tensor<2x2xf32>
// CHECK: %[[I:.*]] = "test.tensor"() : () -> tensor<2x2xf32>
// CHECK: %[[O:.*]] = "test.tensor"() : () -> tensor<2x2xf32>
// CHECK: %[[OP:.*]] = "rock.conv_bwd_weight"(%[[F]], %[[I]], %[[O]]) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[OUT:.*]] = "test.get_out_argument"(%[[OP]]) : (!rock.conv_bwd_weight) -> tensor<2x2xf32>
// CHECK: return %[[OUT]]
func.func @get_out_argument_basic() -> tensor<2x2xf32> {
  %filter = "test.tensor"() : () -> tensor<2x2xf32>
  %input = "test.tensor"() : () -> tensor<2x2xf32>
  %output = "test.tensor"() : () -> tensor<2x2xf32>
  %op = "rock.conv_bwd_weight"(%filter, %input, %output) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %out = "test.get_out_argument"(%op) : (!rock.conv_bwd_weight) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}

//===----------------------------------------------------------------------===//
// Test: ConvBwdWeightOp::getOutArgument returns correct operand even with different types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @get_out_argument_types
// CHECK: %[[F:.*]] = "test.tensor"() : () -> tensor<4x4xi8>
// CHECK: %[[I:.*]] = "test.tensor"() : () -> tensor<4x4xf16>
// CHECK: %[[O:.*]] = "test.tensor"() : () -> tensor<4x4xf32>
// CHECK: %[[OP:.*]] = "rock.conv_bwd_weight"(%[[F]], %[[I]], %[[O]]) : (tensor<4x4xi8>, tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[OUT:.*]] = "test.get_out_argument"(%[[OP]]) : (!rock.conv_bwd_weight) -> tensor<4x4xi8>
// CHECK: return %[[OUT]]
func.func @get_out_argument_types() -> tensor<4x4xi8> {
  %filter = "test.tensor"() : () -> tensor<4x4xi8>
  %input = "test.tensor"() : () -> tensor<4x4xf16>
  %output = "test.tensor"() : () -> tensor<4x4xf32>
  %op = "rock.conv_bwd_weight"(%filter, %input, %output) : (tensor<4x4xi8>, tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %out = "test.get_out_argument"(%op) : (!rock.conv_bwd_weight) -> tensor<4x4xi8>
  return %out : tensor<4x4xi8>
}

//===----------------------------------------------------------------------===//
// Test: ConvBwdWeightOp::getOutArgument with additional attributes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @get_out_argument_with_attrs
// CHECK: %[[F:.*]] = "test.tensor"() : () -> tensor<1x1xf32>
// CHECK: %[[I:.*]] = "test.tensor"() : () -> tensor<1x1xf32>
// CHECK: %[[O:.*]] = "test.tensor"() : () -> tensor<1x1xf32>
// CHECK: %[[OP:.*]] = "rock.conv_bwd_weight"(%[[F]], %[[I]], %[[O]]) {attr1 = 42 : i32} : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[OUT:.*]] = "test.get_out_argument"(%[[OP]]) : (!rock.conv_bwd_weight) -> tensor<1x1xf32>
// CHECK: return %[[OUT]]
func.func @get_out_argument_with_attrs() -> tensor<1x1xf32> {
  %filter = "test.tensor"() : () -> tensor<1x1xf32>
  %input = "test.tensor"() : () -> tensor<1x1xf32>
  %output = "test.tensor"() : () -> tensor<1x1xf32>
  %op = "rock.conv_bwd_weight"(%filter, %input, %output) {attr1 = 42 : i32} : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  %out = "test.get_out_argument"(%op) : (!rock.conv_bwd_weight) -> tensor<1x1xf32>
  return %out : tensor<1x1xf32>
}