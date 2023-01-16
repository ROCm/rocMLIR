// RUN: rocmlir-opt -convert-rock-to-gpu %s | FileCheck %s
// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: memref.global "private" constant @const : memref<2xi32> = dense<[[VALUE:.*]]>
memref.global "private" constant @const : memref<2xi32> = dense<"0xDEADBEEFBEEFDEAD">
// CHECK: gpu.module @kern_module
// CHECK-NEXT: gpu.func @kern
func.func @kern(%arg0 : memref<2xi32>) attributes {kernel, block_size = 64 : i32, grid_size = 1 : i32 } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.get_global @const : memref<2xi32>
  %1 = memref.load %0[%c0] : memref<2xi32>
  %2 = memref.load %0[%c1] : memref<2xi32>
  memref.store %1, %arg0[%c1] : memref<2xi32>
  memref.store %2, %arg0[%c0] : memref<2xi32>
  func.return
}
// CHECK: memref.global "private" constant @const : memref<2xi32> = dense<[[VALUE]]>
// CHECK-NEXT: }
// CHECK-NEXT: }
