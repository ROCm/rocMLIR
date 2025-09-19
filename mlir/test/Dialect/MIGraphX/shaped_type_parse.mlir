// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: scalar (no shape, no stride)
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @scalar
  func.func @scalar() -> !migraphx.shaped<f32> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
    return %0 : !migraphx.shaped<f32>
  }
}

//===----------------------------------------------------------------------===//
// Valid: static shape, static stride
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @static_shape_stride
  func.func @static_shape_stride() -> !migraphx.shaped<2x3xf32, 6x2x1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
    return %0 : !migraphx.shaped<2x3xf32, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: dynamic shape, static stride
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @dynamic_shape
  func.func @dynamic_shape() -> !migraphx.shaped<?x3xf32, 6x2x1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<?x3xf32, 6x2x1>
    return %0 : !migraphx.shaped<?x3xf32, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: static shape, dynamic stride
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @dynamic_stride
  func.func @dynamic_stride() -> !migraphx.shaped<2x3xf32, ?x2x?> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, ?x2x?>
    return %0 : !migraphx.shaped<2x3xf32, ?x2x?>
  }
}

//===----------------------------------------------------------------------===//
// Valid: dynamic shape, dynamic stride
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @dynamic_shape_stride
  func.func @dynamic_shape_stride() -> !migraphx.shaped<?x?xf32, ?x?> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<?x?xf32, ?x?>
    return %0 : !migraphx.shaped<?x?xf32, ?x?>
  }
}

//===----------------------------------------------------------------------===//
// Valid: integer element type
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @int_type
  func.func @int_type() -> !migraphx.shaped<2x2xi32, 2x1> {
    %0 = "migraphx.literal"() {value = dense<1> : tensor<2x2xi32>} : () -> !migraphx.shaped<2x2xi32, 2x1>
    return %0 : !migraphx.shaped<2x2xi32, 2x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: bool element type
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @bool_type
  func.func @bool_type() -> !migraphx.shaped<4xi1, 1> {
    %0 = "migraphx.literal"() {value = dense<true> : tensor<4xi1>} : () -> !migraphx.shaped<4xi1, 1>
    return %0 : !migraphx.shaped<4xi1, 1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: shape with one dimension
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @one_dim
  func.func @one_dim() -> !migraphx.shaped<5xf32, 1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<5xf32>} : () -> !migraphx.shaped<5xf32, 1>
    return %0 : !migraphx.shaped<5xf32, 1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: missing '<'
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{expected shaped dimension list with type}}
  func.func @missing_lt() -> !migraphx.shaped2x3xf32, 6x2x1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped2x3xf32, 6x2x1>
    return %0 : !migraphx.shaped2x3xf32, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: missing '>'
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{expected `>`}}
  func.func @missing_gt() -> !migraphx.shaped<2x3xf32, 6x2x1 {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1
    return %0 : !migraphx.shaped<2x3xf32, 6x2x1
  }
}

//===----------------------------------------------------------------------===//
// Invalid: missing type
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{expected shaped dimension list with type}}
  func.func @missing_type() -> !migraphx.shaped<2x3, 6x2x1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3, 6x2x1>
    return %0 : !migraphx.shaped<2x3, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: missing comma before stride
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{expected `,` and a `x`-separated list in non-scalar migraphx.shaped type}}
  func.func @missing_comma_stride() -> !migraphx.shaped<2x3xf32 6x2x1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32 6x2x1>
    return %0 : !migraphx.shaped<2x3xf32 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: malformed element type
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{expected shaped dimension list with type}}
  func.func @malformed_type() -> !migraphx.shaped<2x3xfoo, 6x2x1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xfoo>} : () -> !migraphx.shaped<2x3xfoo, 6x2x1>
    return %0 : !migraphx.shaped<2x3xfoo, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: stray comma (scalar)
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{expected `,` and a `x`-separated list in non-scalar migraphx.shaped type}}
  func.func @scalar_stray_comma() -> !migraphx.shaped<f32,> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32,>
    return %0 : !migraphx.shaped<f32,>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: empty shape and stride (should parse as scalar, but test for robustness)
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @empty_shape_stride
  func.func @empty_shape_stride() -> !migraphx.shaped<f32> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
    return %0 : !migraphx.shaped<f32>
  }
}