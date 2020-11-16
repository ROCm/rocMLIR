// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s
func @letao_conv2d(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  letao.conv2d(%filter, %input, %output) {
    filter_layout = ["k", "c", "y", "x"],
    input_layout = ["n", "c", "hi", "wi"],
    output_layout = ["n", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0]
  } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}

// CHECK-LABEL: func @letao_conv2d
// CHECK-NEXT: letao.conv2d

func @letao_movepos(%buffer_f32 : memref<2xf32, 5>, %buffer_i32 : memref<2xi32, 5>) {
  %deltaY_i32 = constant 16 : i32
  %deltaX_i32 = constant 8 : i32
  letao.movepos(%buffer_i32, %deltaY_i32, %deltaX_i32) : memref<2xi32, 5>

  %deltaY_f32 = constant 16.0 : f32
  %deltaX_f32 = constant 8.0 : f32
  letao.movepos(%buffer_f32, %deltaY_f32, %deltaX_f32) : memref<2xf32, 5>

  return
}

// CHECK-LABEL: func @letao_movepos
//   CHECK: letao.movepos
//   CHECK: letao.movepos