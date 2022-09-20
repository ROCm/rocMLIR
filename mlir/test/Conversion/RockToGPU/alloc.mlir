// RUN: rocmlir-opt -convert-rock-to-gpu %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @allockernel_module
// CHECK-NEXT: gpu.func @allockernel(%{{.*}}: memref<?x?x?x?xf32>) workgroup(%{{.*}}: memref<16xf32, 3>) private(%{{.*}}: memref<16xf32, 5>) kernel
module {
  func.func @allockernel(%arg0: memref<?x?x?x?xf32>) attributes {kernel = 0 : i32} {
    %cst = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %buffer_lds = rock.alloc() : memref<16xf32, 3>
    %buffer_vgpr = rock.alloc() : memref<16xf32, 5>
    memref.store %cst, %buffer_lds[%c0] : memref<16xf32, 3>
    memref.store %cst, %buffer_vgpr[%c0] : memref<16xf32, 5>
    return
  }
}
