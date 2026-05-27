// RUN: rocmlir-opt -convert-rock-to-gpu %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK-NEXT: gpu.module @emptykernel_module
// CHECK-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32> {llvm.noalias}) kernel
// CHECK-SAME: rock.arch = "gfx90a"
module {
  func.func @emptykernel(%arg0: memref<?x?x?x?xf32> {llvm.noalias}) attributes {rock.kernel = 0 : i32, block_size = 32 : i32, grid_size = 1 : i32, rock.arch = "gfx90a"} {
    return
  }
}
