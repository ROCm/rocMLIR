// RUN: miopen-opt -allow-unregistered-dialect %s | miopen-opt -convert-gpu-to-rocdl |FileCheck %s
module attributes {gpu.container_module} {
  gpu.module @cast {
// CHECK-LABEL: llvm.func @cast
// CHECK: llvm.mlir.constant
// CHECK: llvm.bitcast
// CHECK: llvm.mlir.constant
// CHECK: llvm.lshr
    gpu.func @cast() -> i16 {
      %0 = arith.constant 3.2 : f32
      %1 = gpu.bf_convert %0  : f32 to i16
      gpu.return %1 : i16
    }
// CHECK-LABEL: llvm.func @cast_vector
// CHECK: llvm.mlir.constant
// CHECK: llvm.bitcast
// CHECK: llvm.mlir.constant
// CHECK: llvm.lshr
    gpu.func @cast_vector() -> vector<4xi16> {
      %0 = arith.constant dense<3.2> : vector<4xf32>
      %1 = gpu.bf_convert %0 : vector<4xf32> to vector<4xi16>
      gpu.return %1 : vector<4xi16>
    }
  }
}
