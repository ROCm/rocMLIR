// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @miopen_conv2d(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  miopen.conv2d
  return
}
// CHECK-LABEL: func @miopen_conv2d
//  CHECK-NEXT: miopen.conv2d

func @miopen_transform(%memref : memref<?x?x?x?xf32>) {
  miopen.transform
  return
}

// CHECK-LABEL: func @miopen_transform
//  CHECK-NEXT: miopen.transform

func @miopen_gridwise_gemm(%A : memref<?x?xf32>, %B : memref<?x?xf32>, %C : memref<?x?xf32>) {
  miopen.gridwise_gemm
  return
}

// CHECK-LABEL: func @miopen_gridwise_gemm
//  CHECK-NEXT: miopen.gridwise_gemm
