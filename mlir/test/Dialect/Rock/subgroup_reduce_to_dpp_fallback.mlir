// Verify that rock-subgroup-reduce-to-dpp lowers gpu.subgroup_reduce ops
// that DPP cannot handle via the shuffle-based fallback.

// RUN: rocmlir-opt --rock-subgroup-reduce-to-dpp="chip=gfx942" %s | FileCheck %s
// RUN: rocmlir-opt --rock-subgroup-reduce-to-dpp="chip=gfx950" %s | FileCheck %s
// RUN: rocmlir-opt --rock-subgroup-reduce-to-dpp="chip=gfx1100" %s | FileCheck %s
// RUN: rocmlir-opt --rock-subgroup-reduce-to-dpp="chip=gfx1201" %s | FileCheck %s

// CHECK-LABEL: gpu.module @test_module
gpu.module @test_module {

  // Strided cluster (stride=2) -- DPP rejects non-contiguous lanes,
  // so this must fall through to the shuffle-based fallback.
  // CHECK-LABEL: gpu.func @test_shuffle_fallback
  // CHECK-NOT: gpu.subgroup_reduce
  // CHECK-NOT: amdgpu.dpp
  // CHECK: gpu.shuffle xor
  gpu.func @test_shuffle_fallback(%val : f32) -> f32 {
    %res = gpu.subgroup_reduce add %val cluster(size = 4, stride = 2) : (f32) -> f32
    gpu.return %res : f32
  }
}
