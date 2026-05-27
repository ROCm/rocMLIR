// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -convert-rock-to-gpu | FileCheck %s

// CHECK: module attributes {gpu.container_module}
// CHECK: gpu.module @emptykernel_module
// CHECK-NEXT: gpu.func @emptykernel(%{{.*}}: memref<?x?x?x?xf32>) kernel
module {
  gpu.module @existing_module {
  }
  func.func @emptykernel(%arg0: memref<?x?x?x?xf32>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = 0 : i32, block_size = 32 : i32, grid_size = 1 : i32} {
    return
  }
}
