// RUN: mlir-opt %s --test-gpu-to-hsaco | FileCheck %s
// RUN: mlir-opt %s --test-gpu-to-hsaco=dump-asm=true 2>&1 |\
// RUN:   FileCheck %s --check-prefix=CHECK-ASM

// CHECK: gpu.module @foo attributes {gpu.binary = "HSACO"}
// CHECK-ASM: .globl kernel
gpu.module @foo {
  llvm.func @kernel(%arg0 : f32, %arg1 : !llvm.ptr<f32>)
    // CHECK: attributes  {gpu.kernel}
    attributes  { gpu.kernel } {
    llvm.return
  }
}

// CHECK: gpu.module @bar attributes {gpu.binary = "HSACO"}
gpu.module @bar {
  // CHECK: func @kernel_a
  llvm.func @kernel_a()
    attributes  { gpu.kernel } {
    llvm.return
  }

  // CHECK: func @kernel_b
  llvm.func @kernel_b()
    attributes  { gpu.kernel } {
    llvm.return
  }
}
// CHECK-ASM: amdhsa.target:
