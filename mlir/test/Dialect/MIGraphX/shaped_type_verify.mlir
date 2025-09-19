// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: shape and strides match, valid element type
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @valid_shape_stride
  func.func @valid_shape_stride() -> !migraphx.shaped<2x3xf32, 6x2x1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
    return %0 : !migraphx.shaped<2x3xf32, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: shape and strides length mismatch
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{migraphx.shaped type has 2 elements in its shape but 3 strides defined}}
  func.func @mismatch_shape_stride() -> !migraphx.shaped<2x3xf32, 6x2x1x1> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1x1>
    return %0 : !migraphx.shaped<2x3xf32, 6x2x1x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: element type not allowed (tuple)
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{cannot put the type tuple<i32, f32> into a migraphx.shaped type}}
  func.func @invalid_element_type() -> !migraphx.shaped<2x2xtuple<i32, f32>, 2x1> {
    %0 = "migraphx.literal"() {value = dense<1> : tensor<2x2xi32>} : () -> !migraphx.shaped<2x2xtuple<i32, f32>, 2x1>
    return %0 : !migraphx.shaped<2x2xtuple<i32, f32>, 2x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: empty shape and stride (scalar)
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @scalar
  func.func @scalar() -> !migraphx.shaped<f32> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<f32>} : () -> !migraphx.shaped<f32>
    return %0 : !migraphx.shaped<f32>
  }
}

//===----------------------------------------------------------------------===//
// Valid: dynamic shape and stride
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @dynamic_shape_stride
  func.func @dynamic_shape_stride() -> !migraphx.shaped<?x?xf32, ?x?> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<?x?xf32, ?x?>
    return %0 : !migraphx.shaped<?x?xf32, ?x?>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: shape and stride mismatch with dynamic
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{migraphx.shaped type has 3 elements in its shape but 2 strides defined}}
  func.func @dynamic_mismatch() -> !migraphx.shaped<?x?x?xf32, ?x?> {
    %0 = "migraphx.literal"() {value = dense<1.0> : tensor<2x3x4xf32>} : () -> !migraphx.shaped<?x?x?xf32, ?x?>
    return %0 : !migraphx.shaped<?x?x?xf32, ?x?>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: element type not allowed (memref)
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{cannot put the type memref<2xf32> into a migraphx.shaped type}}
  func.func @invalid_memref_element() -> !migraphx.shaped<2x2xmemref<2xf32>, 2x1> {
    %0 = "migraphx.literal"() {value = dense<1> : tensor<2x2xi32>} : () -> !migraphx.shaped<2x2xmemref<2xf32>, 2x1>
    return %0 : !migraphx.shaped<2x2xmemref<2xf32>, 2x1>
  }
}