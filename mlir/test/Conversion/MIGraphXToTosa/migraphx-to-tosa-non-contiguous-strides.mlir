// RUN: rocmlir-opt -split-input-file --migraphx-to-tosa -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @mlir_dot_sigmoid
func.func @mlir_dot_sigmoid(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x24x1> attributes {kernel = "mixr"} {
  // CHECK: tosa.matmul
  // CHECK-SAME: -> tensor<4x24x24xf16>
  %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
  // CHECK: %[[SIGMOID:.*]] = tosa.sigmoid
  // CHECK-SAME: -> tensor<4x24x24xf16>
  %1 = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x24x1>
  // CHECK: tosa.custom %[[SIGMOID]] {domain_name = "rocmlir", implementation_attrs = "", operator_name = "expand_strides"}
  // CHECK-SAME: (tensor<4x24x24xf16>) -> tensor<4x48x24xf16>
  // CHECK: tosa.reshape
  // CHECK-SAME: -> tensor<4608xf16>
  return %1 : !migraphx.shaped<4x24x24xf16, 1152x24x1>
}

// -----

// expected-error @unknown {{!migraphx.shaped type with smallest stride 2 has no supported in-memory layout}}
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @no_unit_stride(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x48x2> attributes {kernel = "mixr"} {
  %0 = migraphx.sigmoid %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x48x2>
  return %0 : !migraphx.shaped<4x24x24xf16, 1152x48x2>
}

// -----

// expected-error @unknown {{!migraphx.shaped type can't be laid out in memory when the stride 1000 at index 0 being smaller than the product of previous lengths 2400}}
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @stride_not_divisible(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 1000x100x1> attributes {kernel = "mixr"} {
  %0 = migraphx.sigmoid %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1000x100x1>
  return %0 : !migraphx.shaped<4x24x24xf16, 1000x100x1>
}

// -----

func.func @write_to_broadcast(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 0x24x1> attributes {kernel = "mixr"} {
  // expected-error @+2 {{'migraphx.mlir.as.underlying.shape' op writing to tensors with broadcasts is unsupported}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.mlir.as.underlying.shape'}}
  %0 = migraphx.sigmoid %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 0x24x1>
  return %0 : !migraphx.shaped<4x24x24xf16, 0x24x1>
}

// -----

// expected-error @unknown {{!migraphx.shaped type can't be laid out in memory when the stride 576 at index 0 does not evenly divide the previous stride 10}}
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @stride_too_small(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 576x10x1> attributes {kernel = "mixr"} {
  %0 = migraphx.sigmoid %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 576x10x1>
  return %0 : !migraphx.shaped<4x24x24xf16, 576x10x1>
}

// -----

func.func @stride_not_multiple(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 1200x24x1> attributes {kernel = "mixr"} {
  // expected-error @+2 {{'migraphx.mlir.as.underlying.shape' op memory layout dimension 50 is not a multiple of logical dimension 24; this indicates invalid strides}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.mlir.as.underlying.shape'}}
  %0 = migraphx.sigmoid %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1200x24x1>
  return %0 : !migraphx.shaped<4x24x24xf16, 1200x24x1>
}

