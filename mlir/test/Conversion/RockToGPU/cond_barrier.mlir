// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -convert-rock-to-gpu | FileCheck %s

// Test rock.cond_barrier lowering: splits block and creates cf.cond_br
// with amdgpu.s_barrier in the true branch.

// CHECK: module attributes {gpu.container_module}
// CHECK: gpu.module @condbarrier_module
// CHECK: gpu.func @condbarrier
module {
  func.func @condbarrier(%pred: i1)
      attributes {kernel = 0 : i32, arch = "##TOKEN_ARCH##", block_size = 64 : i32, grid_size = 1 : i32} {
    // The cond_barrier lowers to: cf.cond_br %pred, ^true, ^merge
    //                              ^true: amdgpu.s_barrier; cf.br ^merge
    //                              ^merge: ...
    // CHECK: cf.cond_br %{{.*}}, ^[[TRUE:bb[0-9]+]], ^[[MERGE:bb[0-9]+]]
    // CHECK: ^[[TRUE]]:
    // CHECK-NEXT: amdgpu.s_barrier
    // CHECK-NEXT: cf.br ^[[MERGE]]
    // CHECK: ^[[MERGE]]:
    // CHECK: gpu.return
    rock.cond_barrier %pred : i1
    return
  }
}
