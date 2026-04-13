// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -convert-rock-to-gpu | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @allockernel_module
// CHECK-NEXT: gpu.func @allockernel(%{{.*}}: memref<?x?x?x?xf32>) workgroup(%{{.*}}: memref<16xf32, #gpu.address_space<workgroup>> {llvm.align = 64 : i64}) private(%{{.*}}: memref<16xf32, #gpu.address_space<private>>) kernel
module {
  func.func @allockernel(%arg0: memref<?x?x?x?xf32>) attributes {rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##", grid_size = 1 : i32, block_size = 32 : i32} {
    %cst = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %buffer_lds = rock.alloc() : memref<16xf32, #gpu.address_space<workgroup>>
    %buffer_vgpr = rock.alloc() : memref<16xf32, #gpu.address_space<private>>
    memref.store %cst, %buffer_lds[%c0] : memref<16xf32, #gpu.address_space<workgroup>>
    memref.store %cst, %buffer_vgpr[%c0] : memref<16xf32, #gpu.address_space<private>>
    return
  }
}
