// RUN:  miopen-opt -convert-gpu-to-rocdl %s | FileCheck %s
module attributes {gpu.container_module} {
  gpu.module @cast {
// CHECK-LABEL: llvm.func @cast
// CHECK: llvm.mlir.constant
// CHECK: llvm.mlir.constant
// CHECK: llvm.bitcast
// CHECK: llvm.lshr
    gpu.func @cast() -> bf16 {
      %0 = arith.constant 3.2 : f32
      %1 = arith.truncf %0  : f32 to bf16
      gpu.return %1 : bf16
    }
// CHECK-LABEL: llvm.func @cast_vector
// CHECK: llvm.mlir.constant
// CHECK: llvm.mlir.constant
// CHECK: llvm.bitcast
// CHECK: llvm.lshr
    gpu.func @cast_vector() -> vector<4xbf16> {
      %0 = arith.constant dense<3.2> : vector<4xf32>
      %1 = arith.truncf %0 : vector<4xf32> to vector<4xbf16>
      gpu.return %1 : vector<4xbf16>
    }
  }
}
