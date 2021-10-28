// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: func @foo
  func @foo() {
    // CHECK-NEXT: %[[ARG:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: llvm.call @mgpuSetDefaultDevice(%[[ARG]]) : (i32) -> ()
    gpu.set_default_device { devIndex = 1 : i32 }
    return
  }
}
