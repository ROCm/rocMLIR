// RUN: miopen-opt -allow-unregistered-dialect %s | miopen-opt -convert-gpu-to-rocdl |FileCheck %s
module attributes {gpu.container_module} {
  gpu.module @cast {
// CHECK-LABEL: llvm.func @cast
// CHECK: llvm.mlir.constant
// CHECK: llvm.bitcast
// CHECK: llvm.mlir.constant
// CHECK: llvm.lshr
    gpu.func @cast() {
      %0 = constant 3.2 : f32
      gpu.bf_convert %0  : f32 to i16
      gpu.return
    }
  }
}
