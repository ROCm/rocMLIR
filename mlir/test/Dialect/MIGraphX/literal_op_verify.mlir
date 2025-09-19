// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: splat literal, any strides
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @splat_literal
  func.func @splat_literal() -> !migraphx.shaped<2x3xf32, 6x2x1> {
    %c = "migraphx.literal"() {value = dense<1.5> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
    return %c : !migraphx.shaped<2x3xf32, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: non-splat literal, standard strides
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @nonsplat_standard
  func.func @nonsplat_standard() -> !migraphx.shaped<2x3xf32, 6x2x1> {
    %c = "migraphx.literal"() {value = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
    return %c : !migraphx.shaped<2x3xf32, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: non-splat literal, shape mismatch
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{non-splat literals must have a value that matches the literal's logical shape}}
  func.func @nonsplat_shape_mismatch() -> !migraphx.shaped<2x3xf32, 6x2x1> {
    %c = "migraphx.literal"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> !migraphx.shaped<2x3xf32, 6x2x1>
    return %c : !migraphx.shaped<2x3xf32, 6x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: non-splat literal, non-standard strides
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{strides of non-splat literal are not in standard shape}}
  func.func @nonsplat_nonstandard_strides() -> !migraphx.shaped<2x3xf32, 2x6x1> {
    %c = "migraphx.literal"() {value = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 2x6x1>
    return %c : !migraphx.shaped<2x3xf32, 2x6x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: splat literal, non-standard strides (allowed)
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @splat_nonstandard_strides
  func.func @splat_nonstandard_strides() -> !migraphx.shaped<2x3xf32, 2x6x1> {
    %c = "migraphx.literal"() {value = dense<0.0> : tensor<2x3xf32>} : () -> !migraphx.shaped<2x3xf32, 2x6x1>
    return %c : !migraphx.shaped<2x3xf32, 2x6x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: non-splat literal, 1D, stride 1
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @nonsplat_1d
  func.func @nonsplat_1d() -> !migraphx.shaped<4xf32, 1> {
    %c = "migraphx.literal"() {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : () -> !migraphx.shaped<4xf32, 1>
    return %c : !migraphx.shaped<4xf32, 1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: non-splat literal, 1D, stride not 1
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{strides of non-splat literal are not in standard shape}}
  func.func @nonsplat_1d_bad_stride() -> !migraphx.shaped<4xf32, 2> {
    %c = "migraphx.literal"() {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : () -> !migraphx.shaped<4xf32, 2>
    return %c : !migraphx.shaped<4xf32, 2>
  }
}

//===----------------------------------------------------------------------===//
// Valid: splat literal, 1D, stride not 1 (allowed)
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @splat_1d_bad_stride
  func.func @splat_1d_bad_stride() -> !migraphx.shaped<4xf32, 2> {
    %c = "migraphx.literal"() {value = dense<7.0> : tensor<4xf32>} : () -> !migraphx.shaped<4xf32, 2>
    return %c : !migraphx.shaped<4xf32, 2>
  }
}

//===----------------------------------------------------------------------===//
// Valid: non-splat literal, 3D, standard strides
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @nonsplat_3d_standard
  func.func @nonsplat_3d_standard() -> !migraphx.shaped<2x2x2xf32, 4x2x1> {
    %c = "migraphx.literal"() {value = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<2x2x2xf32>} : () -> !migraphx.shaped<2x2x2xf32, 4x2x1>
    return %c : !migraphx.shaped<2x2x2xf32, 4x2x1>
  }
}

//===----------------------------------------------------------------------===//
// Invalid: non-splat literal, 3D, non-standard strides
//===----------------------------------------------------------------------===//
module {
  // expected-error @+1 {{strides of non-splat literal are not in standard shape}}
  func.func @nonsplat_3d_nonstandard() -> !migraphx.shaped<2x2x2xf32, 2x4x1> {
    %c = "migraphx.literal"() {value = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<2x2x2xf32>} : () -> !migraphx.shaped<2x2x2xf32, 2x4x1>
    return %c : !migraphx.shaped<2x2x2xf32, 2x4x1>
  }
}

//===----------------------------------------------------------------------===//
// Valid: splat literal, 3D, non-standard strides (allowed)
//===----------------------------------------------------------------------===//
module {
  // CHECK-LABEL: func @splat_3d_nonstandard
  func.func @splat_3d_nonstandard() -> !migraphx.shaped<2x2x2xf32, 2x4x1> {
    %c = "migraphx.literal"() {value = dense<9.0> : tensor<2x2x2xf32>} : () -> !migraphx.shaped<2x2x2xf32, 2x4x1>
    return %c : !migraphx.shaped<2x2x2xf32, 2x4x1>
  }
}