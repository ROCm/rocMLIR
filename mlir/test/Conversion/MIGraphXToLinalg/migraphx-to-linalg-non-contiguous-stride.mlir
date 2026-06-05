// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @mlir_dot_log(
func.func @mlir_dot_log(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x24x1> attributes {kernel = "mixr"} {
  // CHECK: linalg.batch_matmul ins
  // CHECK-SAME: -> tensor<4x24x24xf16>
  %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
  // CHECK: %[[LOG:.*]] = linalg.log ins{{.*}} -> tensor<4x24x24xf16>
  %1 = migraphx.log %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x24x1>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x48x24xf16>
  // CHECK: tensor.insert_slice %[[LOG]] into %[[EMPTY]]
  return %1 : !migraphx.shaped<4x24x24xf16, 1152x24x1>
}

// -----

// CHECK-LABEL: func.func @mlir_dot_log
func.func @mlir_dot_log(%arg0: !migraphx.shaped<4x5x16xf16, 80x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x5x24xf16, 288x24x1> attributes {arch = "gfx1201", kernel = "mixr", num_cu = 32 : i64} {
  // CHECK: linalg.batch_matmul ins{{.*}}-> tensor<4x5x24xf16>
  %0 = migraphx.dot %arg0, %arg1 : <4x5x16xf16, 80x16x1>, <4x16x24xf16, 384x24x1> -> <4x5x24xf16, 120x24x1>
  // CHECK: %[[LOG:.*]] = linalg.log ins{{.*}}-> tensor<4x5x24xf16>
  %1 = migraphx.log %0 : <4x5x24xf16, 120x24x1> -> <4x5x24xf16, 288x24x1>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x12x24xf16>
  // CHECK: tensor.insert_slice %[[LOG]] into %[[EMPTY]]
  return %1 : !migraphx.shaped<4x5x24xf16, 288x24x1>
}

// -----

// expected-error @unknown {{!migraphx.shaped type with smallest stride 2 has no supported in-memory layout}}
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @no_unit_stride(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x48x2> attributes {kernel = "mixr"} {
  %0 = migraphx.log %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x48x2>
  return %0 : !migraphx.shaped<4x24x24xf16, 1152x48x2>
}

// -----

// expected-error @unknown {{!migraphx.shaped type can't be laid out in memory when the stride 1000 at index 0 being smaller than the product of previous lengths 2400}}
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @stride_not_divisible(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 1000x100x1> attributes {kernel = "mixr"} {
  %0 = migraphx.log %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1000x100x1>
  return %0 : !migraphx.shaped<4x24x24xf16, 1000x100x1>
}

// -----

func.func @write_to_broadcast(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 0x24x1> attributes {kernel = "mixr"} {
  // expected-error @+2 {{'migraphx.mlir.as.underlying.shape' op writing to tensors with broadcasts is unsupported}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.mlir.as.underlying.shape'}}
  %0 = migraphx.log %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 0x24x1>
  return %0 : !migraphx.shaped<4x24x24xf16, 0x24x1>
}

// -----

// expected-error @unknown {{!migraphx.shaped type can't be laid out in memory when the stride 576 at index 0 does not evenly divide the previous stride 10}}
// expected-error @below {{failed to legalize operation 'func.func'}}
func.func @stride_too_small(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 576x10x1> attributes {kernel = "mixr"} {
  %0 = migraphx.log %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 576x10x1>
  return %0 : !migraphx.shaped<4x24x24xf16, 576x10x1>
}
