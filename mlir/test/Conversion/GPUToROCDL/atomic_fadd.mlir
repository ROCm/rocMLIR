// RUN: miopen-opt %s -convert-gpu-to-rocdl | FileCheck %s

gpu.module @atomic_fadd {
  // f32 tests.

  // CHECK-LABEL: llvm.func @atomic_fadd_f32_to_rank_1
  gpu.func @atomic_fadd_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %offset0 : i32) {
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.atomic_fadd(%value, %dst, %offset0) : f32, memref<128xf32>, i32
    gpu.return
  }

  // CHECK-LABEL: llvm.func @atomic_fadd_f32_to_rank_4
  gpu.func @atomic_fadd_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.atomic_fadd(%value, %dst, %offset0, %offset1, %offset2, %offset3) : f32, memref<128x64x32x16xf32>, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: llvm.func @atomic_fadd_2xf32_to_rank_1
  gpu.func @atomic_fadd_2xf32_to_rank_1(%value : vector<2xf32>, %dst : memref<128xf32>, %offset0 : i32) {
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.atomic_fadd(%value, %dst, %offset0) : vector<2xf32>, memref<128xf32>, i32
    gpu.return
  }

  // CHECK-LABEL: llvm.func @atomic_fadd_2xf32_to_rank_4
  gpu.func @atomic_fadd_2xf32_to_rank_4(%value : vector<2xf32>, %dst : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.atomic_fadd(%value, %dst, %offset0, %offset1, %offset2, %offset3) : vector<2xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: llvm.func @atomic_fadd_4xf32_to_rank_1
  gpu.func @atomic_fadd_4xf32_to_rank_1(%value : vector<4xf32>, %dst : memref<128xf32>, %offset0 : i32) {
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.atomic_fadd(%value, %dst, %offset0) : vector<4xf32>, memref<128xf32>, i32
    gpu.return
  }

  // CHECK-LABEL: llvm.func @atomic_fadd_4xf32_to_rank_4
  gpu.func @atomic_fadd_4xf32_to_rank_4(%value : vector<4xf32>, %dst : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK: rocdl.atomic.fadd %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.atomic_fadd(%value, %dst, %offset0, %offset1, %offset2, %offset3) : vector<4xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32
    gpu.return
  }
}
